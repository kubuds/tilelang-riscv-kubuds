/*
 * RISC-V kernel harness for TileLang cross-compiled kernels.
 *
 * Purpose
 * -------
 * TileLang compiles TileLang/TIR kernels to a TVM FFI shared library.  When the
 * kernel is cross-compiled for RISC-V on a non-RISC-V build machine, the
 * library cannot be loaded in the build process, so correctness testing runs
 * this small harness under QEMU user-mode emulation (qemu-riscv64).  The same
 * harness is also handy on real silicon (SG2042 / SpacemiT K3) to smoke-test
 * exported kernels without a Python stack.
 *
 * The harness links against the TVM FFI runtime (tvm-ffi) which provides the
 * module loader and packed-function ABI used by the exported kernel library.
 *
 * Wire protocol (little-endian, stdin -> stdout)
 * ---------------------------------------------
 *   request  := "TLR1" int32:num_args arg*
 *   arg      := int32:kind
 *               kind==0 (scalar int64):   int64:value
 *               kind==2 (scalar float64): float64:value
 *               kind==1 (tensor):         int32:io_flags int32:dtype_code
 *                                         int32:dtype_bits int32:dtype_lanes
 *                                         int32:ndim int64:dims[ndim]
 *   io_flags := bit0 = payload sent to the kernel (input)
 *               bit1 = payload copied back after the call (output)
 *   payload  := raw row-major bytes of every is_input tensor, in arg order
 *   response := int32:status (0 == OK) payload'
 *   payload' := raw row-major bytes of every !is_input tensor, in arg order
 *
 * License: Apache-2.0 (same as the TVM/tvm-ffi code it links against; this
 * file is original code written for TileLang).
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

namespace {

constexpr char kMagic[4] = {'T', 'L', 'R', '1'};

struct TensorArg {
  int io_flags;  // bit0: input payload, bit1: output payload
  DLDataType dtype;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::vector<uint8_t> data;
  DLTensor dl_tensor{};

  bool IsInput() const { return (io_flags & 1) != 0; }
  bool IsOutput() const { return (io_flags & 2) != 0; }

  size_t Numel() const {
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t ItemSize() const { return (dtype.bits * dtype.lanes + 7) / 8; }
  size_t NumBytes() const { return Numel() * ItemSize(); }

  // Compact row-major strides, the layout the packed ABI checks against.
  void ComputeCompactStrides() {
    strides.assign(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
      strides[i - 2] = strides[i - 1] * shape[i - 1];
    }
  }
};

bool ReadExact(FILE* fp, void* dst, size_t nbytes) { return fread(dst, 1, nbytes, fp) == nbytes; }

bool WriteExact(FILE* fp, const void* src, size_t nbytes) { return fwrite(src, 1, nbytes, fp) == nbytes; }

[[noreturn]] void Fail(const char* what) {
  fprintf(stderr, "riscv_kernel_harness: %s\n", what);
  // Keep the response channel well-formed even on failure.
  int32_t status = -1;
  fwrite(&status, sizeof(status), 1, stdout);
  fflush(stdout);
  exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <kernel.so> <function-name>\n", argv[0]);
    return 1;
  }
  const char* lib_path = argv[1];
  const char* func_name = argv[2];

  FILE* in = stdin;
  FILE* out = stdout;

  char magic[4];
  if (!ReadExact(in, magic, 4) || memcmp(magic, kMagic, 4) != 0) Fail("bad request magic");
  int32_t num_args = 0;
  if (!ReadExact(in, &num_args, sizeof(num_args)) || num_args < 0) Fail("bad argument count");

  std::vector<TensorArg> tensors;
  std::vector<int64_t> int_scalars;
  std::vector<double> float_scalars;
  std::vector<int32_t> arg_kinds(static_cast<size_t>(num_args));

  for (int32_t i = 0; i < num_args; ++i) {
    int32_t kind = 0;
    if (!ReadExact(in, &kind, sizeof(kind))) Fail("truncated argument descriptor");
    arg_kinds[static_cast<size_t>(i)] = kind;
    if (kind == 0) {
      int64_t value = 0;
      if (!ReadExact(in, &value, sizeof(value))) Fail("truncated scalar argument");
      int_scalars.push_back(value);
    } else if (kind == 2) {
      double value = 0;
      if (!ReadExact(in, &value, sizeof(value))) Fail("truncated float scalar argument");
      float_scalars.push_back(value);
    } else if (kind == 1) {
      TensorArg t;
      int32_t io_flags = 0, code = 0, bits = 0, lanes = 0, ndim = 0;
      if (!ReadExact(in, &io_flags, sizeof(io_flags)) || !ReadExact(in, &code, sizeof(code)) ||
          !ReadExact(in, &bits, sizeof(bits)) || !ReadExact(in, &lanes, sizeof(lanes)) ||
          !ReadExact(in, &ndim, sizeof(ndim)) || ndim < 0 || ndim > 8) {
        Fail("truncated tensor descriptor");
      }
      t.io_flags = io_flags;
      t.dtype.code = static_cast<uint8_t>(code);
      t.dtype.bits = static_cast<uint8_t>(bits);
      t.dtype.lanes = static_cast<uint16_t>(lanes);
      t.shape.resize(static_cast<size_t>(ndim));
      if (ndim > 0 && !ReadExact(in, t.shape.data(), sizeof(int64_t) * static_cast<size_t>(ndim))) {
        Fail("truncated tensor shape");
      }
      tensors.push_back(std::move(t));
    } else {
      Fail("unknown argument kind");
    }
  }

  // Payload: raw bytes of every input tensor, in declaration order.  All
  // buffers are allocated here (zero-filled) so kernels always receive valid
  // data pointers.
  for (auto& t : tensors) {
    t.data.resize(t.NumBytes(), 0);
    if (!t.IsInput()) continue;
    if (t.NumBytes() > 0 && !ReadExact(in, t.data.data(), t.NumBytes())) Fail("truncated tensor payload");
  }

  // ---- set up DLTensor views ----
  for (auto& t : tensors) {
    t.ComputeCompactStrides();
    t.dl_tensor.data = t.data.empty() ? nullptr : t.data.data();
    t.dl_tensor.device = DLDevice{kDLCPU, 0};
    t.dl_tensor.ndim = static_cast<int>(t.shape.size());
    t.dl_tensor.dtype = t.dtype;
    t.dl_tensor.shape = t.shape.empty() ? nullptr : t.shape.data();
    t.dl_tensor.strides = t.strides.empty() ? nullptr : t.strides.data();
    t.dl_tensor.byte_offset = 0;
  }

  // ---- load the kernel library and call the packed function ----
  int32_t status = 0;
  try {
    tvm::ffi::Module mod = tvm::ffi::Module::LoadFromFile(lib_path);
    tvm::ffi::Optional<tvm::ffi::Function> func = mod->GetFunction(func_name);
    if (!func.defined()) Fail("function not found in kernel library");

    std::vector<tvm::ffi::AnyView> ffi_args;
    ffi_args.reserve(static_cast<size_t>(num_args));
    size_t tensor_cursor = 0;
    size_t int_cursor = 0;
    size_t float_cursor = 0;
    for (int32_t i = 0; i < num_args; ++i) {
      int32_t kind = arg_kinds[static_cast<size_t>(i)];
      if (kind == 0) {
        ffi_args.emplace_back(int_scalars[int_cursor++]);
      } else if (kind == 2) {
        ffi_args.emplace_back(float_scalars[float_cursor++]);
      } else {
        ffi_args.emplace_back(static_cast<DLTensor*>(&tensors[tensor_cursor++].dl_tensor));
      }
    }

    tvm::ffi::Any result;
    (*func)->CallPacked(ffi_args.data(), num_args, &result);
  } catch (const tvm::ffi::Error& err) {
    // Kernel-side errors land here with the compiler/runtime diagnostic.
    status = 1;
    fprintf(stderr, "kernel call failed: %s\n", err.what());
  } catch (const std::exception& err) {
    status = 1;
    fprintf(stderr, "kernel call failed: %s\n", err.what());
  }

  // ---- response: status, then raw bytes of every output tensor ----
  if (!WriteExact(out, &status, sizeof(status))) Fail("cannot write status");
  if (status == 0) {
    for (const auto& t : tensors) {
      if (!t.IsOutput()) continue;
      if (t.NumBytes() > 0 && !WriteExact(out, t.data.data(), t.NumBytes())) Fail("cannot write output payload");
    }
  }
  fflush(out);
  return status == 0 ? 0 : 3;
}
