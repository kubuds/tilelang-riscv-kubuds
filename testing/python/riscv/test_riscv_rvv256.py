"""Phase-2: RVV 1.0 (VLEN=256) kernels for the SpacemiT K3 X100.

The X100 core cluster implements the RISC-V Vector Extension v1.0 with
``VLEN=256`` (see the SpacemiT K3 paper: two arithmetic and two load/store
vector pipelines, ELEN=64).  These tests compile TileLang kernels with the
``k3-x100`` preset and execute them under QEMU with a 256-bit vector length:

* numerical correctness against numpy references,
* a codegen check that the generated machine code really contains RVV
  instructions (so the "RVV 1.0 adaptation" is verifiable, not accidental).
"""

import numpy as np
import pytest

import tilelang
import tilelang.language as T
import tilelang.testing

from conftest import export_kernel, np_assert_close, requires_qemu_stack, run_qemu
from tilelang.riscv.qemu import QEMUTensor

X100_TARGET = "k3-x100"
X100_VLEN = 256

# RISC-V OP-V major opcode (1010111): vsetvl*/vector arithmetic.  Every RVV
# kernel we generate contains at least one of these, so scanning instruction
# words for the opcode is a disassembler-independent way to prove that vector
# code was emitted (GNU binutils may print undecodable RVV words as `.insn`).
_RVV_OP_V_OPCODE = 0x57


def _disassemble_riscv(lib_path) -> str:
    import os
    import shutil
    import subprocess

    objdump = os.environ.get("TILELANG_RISCV_OBJDUMP") or shutil.which("riscv64-linux-gnu-objdump") or shutil.which("llvm-objdump")
    assert objdump is not None, "riscv64 objdump not available for codegen verification"
    args = [objdump, "-d", str(lib_path)]
    if objdump.endswith("llvm-objdump"):
        args = [objdump, "-d", "--mattr=+v", str(lib_path)]
    out = subprocess.run(args, capture_output=True, text=True, check=True)
    return out.stdout


def _has_rvv_code(lib_path, func_symbol: str) -> bool:
    import re

    listing = _disassemble_riscv(lib_path)
    in_function = False
    for line in listing.splitlines():
        header = re.match(r"^\s*[0-9a-f]+\s+<([^>]+)>", line)
        if header:
            in_function = header.group(1) == func_symbol
            continue
        if not in_function:
            continue
        # "  7e0:\t57 70 04 cd  \tvsetivli ..." or "  7e0:\tcd727057 \t.insn ..."
        match = re.match(r"^\s*[0-9a-f]+:\s+((?:[0-9a-f]{2}\s+)+|[0-9a-f]{8})", line)
        if not match:
            continue
        tokens = match.group(1).split()
        if len(tokens) == 1 and len(tokens[0]) == 8:
            word = int(tokens[0], 16)
        elif len(tokens) == 4:
            word = int("".join(reversed(tokens)), 16)
        else:
            continue
        if word & 0x7F == _RVV_OP_V_OPCODE:
            return True
    return False


@requires_qemu_stack
def test_vector_add_f32_rvv256(tmp_path):
    N = 1024  # a whole number of vector registers keeps the vectorized loop tail-free

    @T.prim_func
    def vadd(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            for i in T.vectorized(N):
                C[i] = A[i] + B[i]

    rng = np.random.default_rng(10)
    a = rng.standard_normal(N).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)

    compiled = tilelang.compile(vadd, target=X100_TARGET, execution_backend="tvm_ffi")
    lib = tmp_path / "vadd_rvv.so"
    run_qemu(compiled, [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(out, False)], lib, vlen=X100_VLEN)
    np_assert_close(out, a + b, rtol=1e-6, atol=1e-6)
    assert _has_rvv_code(lib, "__tvm_ffi_vadd"), "expected RVV instructions in the generated code"


@requires_qemu_stack
def test_rv64gc_baseline_has_no_rvv_code(tmp_path):
    """Negative control: the SG2042/RV64GC preset must not emit RVV code.

    Also guards the codegen check itself against matching symbol names instead
    of instructions.
    """

    N = 256

    @T.prim_func
    def vadd(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            for i in T.vectorized(N):
                C[i] = A[i] + B[i]

    compiled = tilelang.compile(vadd, target="sg2042", execution_backend="tvm_ffi")
    lib = tmp_path / "vadd_rv64gc.so"
    export_kernel(compiled, lib)
    assert not _has_rvv_code(lib, "__tvm_ffi_vadd"), "RV64GC baseline must not contain RVV instructions"


@requires_qemu_stack
def test_parallel_loop_autovectorized_rvv256(tmp_path):
    """T.Parallel elementwise math must vectorize and stay correct on RVV."""
    N = 4099

    @T.prim_func
    def poly(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            for i in T.Parallel(N):
                C[i] = A[i] * B[i] + 0.5 * A[i] - B[i]

    rng = np.random.default_rng(11)
    a = rng.standard_normal(N).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)

    compiled = tilelang.compile(poly, target=X100_TARGET, execution_backend="tvm_ffi")
    lib = tmp_path / "poly_rvv.so"
    run_qemu(compiled, [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(out, False)], lib, vlen=X100_VLEN)
    np_assert_close(out, a * b + 0.5 * a - b, rtol=1e-5, atol=1e-5)
    assert _has_rvv_code(lib, "__tvm_ffi_poly"), "expected RVV instructions in the generated code"


@requires_qemu_stack
def test_matmul_f32_rvv256(tmp_path):
    M, N, K = 64, 64, 64

    @T.prim_func
    def matmul(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1):
            for i in T.serial(M):
                for j in T.serial(N):
                    C[i, j] = 0.0
            for i in T.serial(M):
                for j in T.serial(N):
                    for k in T.serial(K):
                        C[i, j] += A[i, k] * B[k, j]

    rng = np.random.default_rng(12)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    out = np.zeros((M, N), dtype=np.float32)

    compiled = tilelang.compile(matmul, target=X100_TARGET, execution_backend="tvm_ffi")
    run_qemu(compiled, [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(out, False)], tmp_path / "matmul_rvv.so", vlen=X100_VLEN)
    np_assert_close(out, a @ b, rtol=1e-3, atol=1e-3)


@requires_qemu_stack
def test_matmul_f16_rvv256(tmp_path):
    M, N, K = 32, 32, 32

    @T.prim_func
    def matmul(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1):
            for i in T.serial(M):
                for j in T.serial(N):
                    C[i, j] = 0.0
            for i in T.serial(M):
                for j in T.serial(N):
                    for k in T.serial(K):
                        C[i, j] += T.cast(A[i, k], "float32") * T.cast(B[k, j], "float32")

    rng = np.random.default_rng(13)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)
    out = np.zeros((M, N), dtype=np.float32)

    compiled = tilelang.compile(matmul, target=X100_TARGET, execution_backend="tvm_ffi")
    run_qemu(compiled, [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(out, False)], tmp_path / "matmul_f16_rvv.so", vlen=X100_VLEN)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np_assert_close(out, expected, rtol=1e-3, atol=1e-3)


@requires_qemu_stack
def test_row_sum_reduce_rvv256(tmp_path):
    M, N = 64, 1024

    @T.prim_func
    def row_sum(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(1):
            for i in T.serial(M):
                B[i] = 0.0
                for j in T.serial(N):
                    B[i] += A[i, j]

    rng = np.random.default_rng(14)
    a = rng.standard_normal((M, N)).astype(np.float32)
    out = np.zeros(M, dtype=np.float32)

    compiled = tilelang.compile(row_sum, target=X100_TARGET, execution_backend="tvm_ffi")
    run_qemu(compiled, [QEMUTensor(a, True), QEMUTensor(out, False)], tmp_path / "row_sum_rvv.so", vlen=X100_VLEN)
    np_assert_close(out, a.sum(axis=1), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tilelang.testing.main()
