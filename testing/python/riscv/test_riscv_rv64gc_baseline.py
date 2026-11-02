"""Phase-1 baseline: pure RV64GC kernels compiled and executed under QEMU.

This is the acceptance test for the scalar RISC-V backend: every kernel is
compiled through the full TileLang -> TIR -> LLVM pipeline for a RV64GC target
(``riscv64`` generic and the SG2042 preset) and then executed under
``qemu-riscv64`` user-mode emulation, comparing against numpy references.

The SG2042 (Sophgo) is the primary test object of this phase: its T-Head C920
cores execute plain RV64GC code (the cores also implement the incompatible
RVV 0.7.1 draft, which is why the SG2042 preset deliberately keeps LLVM vector
codegen disabled).
"""

import numpy as np
import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.riscv.qemu import QEMUTensor

from conftest import np_assert_close, requires_qemu_stack, run_qemu

BASELINE_TARGETS = ["riscv64", "sg2042"]


@requires_qemu_stack
@pytest.mark.parametrize("target", BASELINE_TARGETS)
def test_vector_add_f32(target, tmp_path):
    N = 321  # deliberately not a multiple of anything

    @T.prim_func
    def vadd(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            for i in T.serial(N):
                C[i] = A[i] + B[i]

    rng = np.random.default_rng(0)
    a = rng.standard_normal(N).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)

    compiled = tilelang.compile(vadd, target=target, execution_backend="tvm_ffi")
    run_qemu(
        compiled,
        [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(out, False)],
        tmp_path / "vadd.so",
    )
    np_assert_close(out, a + b, rtol=1e-6, atol=1e-6)


@requires_qemu_stack
@pytest.mark.parametrize("target", BASELINE_TARGETS)
def test_matmul_f32(target, tmp_path):
    M, N, K = 33, 29, 17

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

    rng = np.random.default_rng(1)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    out = np.zeros((M, N), dtype=np.float32)

    compiled = tilelang.compile(matmul, target=target, execution_backend="tvm_ffi")
    run_qemu(
        compiled,
        [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(out, False)],
        tmp_path / "matmul.so",
    )
    np_assert_close(out, a @ b, rtol=1e-4, atol=1e-4)


@requires_qemu_stack
@pytest.mark.parametrize("target", BASELINE_TARGETS)
def test_matmul_f16_accum_f32(target, tmp_path):
    M, N, K = 17, 19, 23

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

    rng = np.random.default_rng(2)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)
    out = np.zeros((M, N), dtype=np.float32)

    compiled = tilelang.compile(matmul, target=target, execution_backend="tvm_ffi")
    run_qemu(
        compiled,
        [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(out, False)],
        tmp_path / "matmul_f16.so",
    )
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np_assert_close(out, expected, rtol=1e-3, atol=1e-3)


@requires_qemu_stack
@pytest.mark.parametrize("target", BASELINE_TARGETS)
def test_row_sum_reduce(target, tmp_path):
    M, N = 13, 4099

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

    rng = np.random.default_rng(3)
    a = rng.standard_normal((M, N)).astype(np.float32)
    out = np.zeros(M, dtype=np.float32)

    compiled = tilelang.compile(row_sum, target=target, execution_backend="tvm_ffi")
    run_qemu(compiled, [QEMUTensor(a, True), QEMUTensor(out, False)], tmp_path / "row_sum.so")
    np_assert_close(out, a.sum(axis=1), rtol=1e-4, atol=1e-4)


@requires_qemu_stack
@pytest.mark.parametrize("target", BASELINE_TARGETS)
def test_scalar_shape_params(target, tmp_path):
    """Integer and floating-point scalar parameters travel the packed ABI."""

    @T.prim_func
    def scale(
        A: T.Tensor((128,), "float32"),
        B: T.Tensor((128,), "float32"),
        n: T.int32,
        s: T.float32,
    ):
        with T.Kernel(1):
            for i in T.serial(128):
                if i < n:
                    B[i] = A[i] * s

    rng = np.random.default_rng(4)
    a = rng.standard_normal(128).astype(np.float32)
    out = np.zeros(128, dtype=np.float32)
    n = 100
    s = 2.5

    compiled = tilelang.compile(scale, target=target, execution_backend="tvm_ffi")
    run_qemu(
        compiled,
        [QEMUTensor(a, True), QEMUTensor(out, False), n, s],
        tmp_path / "scale.so",
    )
    expected = np.zeros(128, dtype=np.float32)
    expected[:n] = a[:n] * np.float32(s)
    np_assert_close(out, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    tilelang.testing.main()
