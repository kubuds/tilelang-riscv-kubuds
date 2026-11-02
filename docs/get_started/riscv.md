# RISC-V Support

TileLang supports RISC-V CPUs through the TileLang + TVM (TIR) route: kernels
are lowered with the TVM TIR pipeline and compiled by TVM's LLVM code
generator for the `riscv64-unknown-linux-gnu` triple. Three deployment
profiles are supported:

| Preset    | Hardware                              | ISA profile |
| --------- | ------------------------------------- | ----------- |
| `riscv64` | any RV64GC Linux machine               | RV64GC (`+m,+a,+f,+d,+c,+zicsr,+zifencei`) |
| `sg2042`  | Sophgo SG2042 (64x T-Head C920)        | RV64GC scalar baseline (see note below) |
| `k3-x100` | SpacemiT K3 X100 core cluster          | RV64GC + RVA23 bit-manip + RVV 1.0, VLEN=256 |
| `k3-a100` | SpacemiT K3 A100 AI-CPU cluster        | RV64GC + RVA23 bit-manip + RVV 1.0, VLEN=1024 + IME-2.0 |

> **Note on the SG2042:** the T-Head C920 cores implement the draft RVV 0.7.1
> vector extension, which is binary-incompatible with RVV 1.0. The `sg2042`
> preset therefore keeps LLVM vector codegen disabled and is the reference
> target for the pure RV64GC baseline.

## Compiling kernels

```python
import tilelang
import tilelang.language as T

@T.prim_func
def vadd(A: T.Tensor((1024,), "float32"), B: T.Tensor((1024,), "float32"),
         C: T.Tensor((1024,), "float32")):
    with T.Kernel(1):
        for i in T.vectorized(1024):
            C[i] = A[i] + B[i]

kernel = tilelang.compile(vadd, target="k3-x100")
```

On a RISC-V host the compiled kernel executes in-process like on any CPU.
Config dict forms and explicit overrides are supported as well:

```python
target = {"kind": "riscv64", "soc": "k3-x100", "mattr": ["+zvfh"]}
```

Plain `llvm` targets with a RISC-V triple are routed to the RISC-V backend
automatically. When `target="auto"` is used on a RISC-V host, the RV64GC
baseline is detected (conservatively: pass `k3-x100`/`k3-a100` explicitly on
vector-capable silicon). `tilelang.riscv.runtime.detect_preset()` can help
pick a preset from the running machine's capabilities.

## Vectorization and VLEN

The loop vectorizer sizes vectors from the target's `+zvl<bits>b` attribute:
one iteration consumes a full vector register (LMUL=1), i.e. 8 lanes of f32
on the X100 (VLEN=256) and 32 lanes on the A100 (VLEN=1024). Generated code
may assume the VLEN it was compiled for; run `k3-a100` builds only on
VLEN>=1024 hardware (or QEMU with `-cpu rv64,v=true,vlen=1024`).

## Cross-compiling and exporting

Cross-compilation works from any host with LLVM-enabled TileLang and a
riscv64 toolchain:

```python
from tilelang.riscv.export import export_riscv_library

kernel = tilelang.compile(vadd, target="sg2042")
export_riscv_library(kernel.adapter.get_exportable_executable(), "vadd.so", target=kernel.target)
```

Toolchain knobs (environment variables):

| Variable | Meaning |
| -------- | ------- |
| `TILELANG_RISCV_GCC` / `TILELANG_RISCV_GXX` | riscv64 gcc driver (default: `riscv64-linux-gnu-gcc` on `PATH`) |
| `TILELANG_RISCV_SYSROOT` | riscv64 sysroot (passed as `--sysroot`) |
| `TILELANG_RISCV_CFLAGS` / `TILELANG_RISCV_LDFLAGS` | extra compile/link flags; library tokens (`-lfoo`, `*.a`) are placed after the objects |

## Testing under QEMU

`src/riscv/harness` contains a small riscv64 runner that loads exported
kernels and invokes them through the TVM-FFI packed-function ABI. It is built
together with the TVM runtime and the IME-2.0 simulator:

```bash
cmake -S src/riscv/harness -B build/riscv-harness \
      -DCMAKE_TOOLCHAIN_FILE=src/riscv/harness/riscv64-linux-gnu.cmake \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build/riscv-harness -j
```

`testing/python/riscv/` runs the correctness suite under `qemu-riscv64`
(suites skip themselves when the toolchain/QEMU/harness are missing).
`tilelang.riscv.qemu.run_kernel()` is the programmatic entry point used by
the tests.

## OpenBLAS GEMM offload

`tilelang.riscv.blas.gemm_blas` lowers a GEMM to a `cblas_sgemm`/
`cblas_dgemm` extern call from TIR, so BLAS micro-kernels fuse with the rest
of the TileLang program. Cross-build OpenBLAS for the X100 with

```bash
make -j CC="riscv64-linux-gnu-gcc -fPIC" HOSTCC=gcc \
     TARGET=RISCV64_ZVL256B NOFORTRAN=1 USE_THREAD=0 NUM_THREADS=1 \
     NO_SHARED=1 BINARY=64
export TILELANG_RISCV_LDFLAGS="-L/path/to/OpenBLAS -lopenblas -lm"
```

(OpenBLAS's RVV kernels need GCC 14+ for the RVV C intrinsics 0.12 API.)

## The SpacemiT IME-2.0 matrix engine (K3 A100)

`tilelang.riscv.ime` exposes the A100's IME-2.0 MMA micro-kernels as TIR
intrinsics (`ime2_mma_i8`: 8x16x8 INT8, `ime2_mma_f16`/`ime2_mma_bf16`:
8x8x8 FP16/BF16, `D = A*B + C`, per the SpacemiT K3 paper). They lower to
extern calls of the `spacemit_ime2` runtime ABI:

```c
void spacemit_ime2_mma_i8(const int8_t* A, const int8_t* B,
                          const int32_t* C, int32_t* D);
void spacemit_ime2_mma_f16(const void* A, const void* B,
                           const float* C, float* D);
void spacemit_ime2_mma_bf16(const void* A, const void* B,
                            const float* C, float* D);
```

`src/riscv/ime2/ime2_sim.cc` implements this ABI as a reference simulator
(the executable specification used for correctness testing under QEMU); on
real silicon the vendor runtime provides the same entry points and issues the
actual IME-2.0 instructions. Link the simulator (or the vendor runtime) into
the exported kernel library via `TILELANG_RISCV_LDFLAGS`, e.g.
`-L build/riscv-harness -lspacemit_ime2_sim`.

## Cache caveat

The kernel disk cache keys include the RISC-V link environment
(`TILELANG_RISCV_*`), so switching toolchains or linked libraries recompiles
instead of reusing stale binaries.
