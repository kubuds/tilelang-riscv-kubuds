"""Phase-2: GEMM offload to OpenBLAS RVV kernels on the SpacemiT K3 X100.

These tests exercise the TIR-level BLAS integration (:mod:`tilelang.riscv.blas`):
a TileLang kernel emits a ``cblas_sgemm`` extern call, the exported riscv64
library links against a cross-built OpenBLAS (``TARGET=RISCV64_ZVL256B``),
and the whole thing runs under QEMU with ``VLEN=256``.

Requires:
* the QEMU execution stack (see conftest.py),
* a cross-built OpenBLAS whose directory is given via
  ``TILELANG_OPENBLAS_LIBDIR`` (must contain ``libopenblas.a`` or
  ``libopenblas.so``).
"""

import os

import numpy as np
import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.riscv import blas
from tilelang.riscv.qemu import QEMUTensor

from conftest import np_assert_close, requires_qemu_stack, run_qemu

OPENBLAS_LIBDIR = os.environ.get("TILELANG_OPENBLAS_LIBDIR")

requires_openblas = pytest.mark.skipif(
    not OPENBLAS_LIBDIR,
    reason="cross-built OpenBLAS not available (set TILELANG_OPENBLAS_LIBDIR)",
)

X100_TARGET = "k3-x100"
X100_VLEN = 256


@requires_qemu_stack
@requires_openblas
@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (True, False), (False, True), (True, True)])
def test_gemm_blas_f32(tmp_path, monkeypatch, transpose_a, transpose_b):
    monkeypatch.setenv("TILELANG_RISCV_LDFLAGS", f"-L{OPENBLAS_LIBDIR} -lopenblas -lm")

    M, N, K = 37, 23, 41
    a_shape = (K, M) if transpose_a else (M, K)
    b_shape = (N, K) if transpose_b else (K, N)

    @T.prim_func
    def gemm(
        A: T.Tensor(a_shape, "float32"),
        B: T.Tensor(b_shape, "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1):
            blas.gemm_blas(A, B, C, M, N, K, transpose_a=transpose_a, transpose_b=transpose_b)

    rng = np.random.default_rng(20)
    a = rng.standard_normal(a_shape).astype(np.float32)
    b = rng.standard_normal(b_shape).astype(np.float32)
    out = np.zeros((M, N), dtype=np.float32)

    compiled = tilelang.compile(gemm, target=X100_TARGET, execution_backend="tvm_ffi")
    run_qemu(
        compiled,
        [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(out, False)],
        tmp_path / f"gemm_blas_{int(transpose_a)}{int(transpose_b)}.so",
        vlen=X100_VLEN,
        extra_env={"OPENBLAS_NUM_THREADS": "1"},
    )
    expected = (a.T if transpose_a else a) @ (b.T if transpose_b else b)
    np_assert_close(out, expected, rtol=1e-3, atol=1e-3)


@requires_qemu_stack
@requires_openblas
def test_gemm_blas_f32_alpha_beta(tmp_path, monkeypatch):
    monkeypatch.setenv("TILELANG_RISCV_LDFLAGS", f"-L{OPENBLAS_LIBDIR} -lopenblas -lm")

    M, N, K = 29, 31, 17
    alpha, beta = 0.75, 0.5

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1):
            blas.gemm_blas(A, B, C, M, N, K, alpha=alpha, beta=beta)

    rng = np.random.default_rng(21)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    c = rng.standard_normal((M, N)).astype(np.float32)
    out = c.copy()

    compiled = tilelang.compile(gemm, target=X100_TARGET, execution_backend="tvm_ffi")
    run_qemu(
        compiled,
        # C is read (scaled by beta) and written: an in/out buffer.
        [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(out, True, is_output=True)],
        tmp_path / "gemm_blas_ab.so",
        vlen=X100_VLEN,
        extra_env={"OPENBLAS_NUM_THREADS": "1"},
    )
    expected = np.float32(alpha) * (a @ b) + np.float32(beta) * c
    np_assert_close(out, expected, rtol=1e-3, atol=1e-3)


@requires_qemu_stack
@requires_openblas
def test_gemm_blas_f64(tmp_path, monkeypatch):
    monkeypatch.setenv("TILELANG_RISCV_LDFLAGS", f"-L{OPENBLAS_LIBDIR} -lopenblas -lm")

    M, N, K = 23, 19, 13

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float64"),
        B: T.Tensor((K, N), "float64"),
        C: T.Tensor((M, N), "float64"),
    ):
        with T.Kernel(1):
            blas.gemm_blas(A, B, C, M, N, K, dtype="float64")

    rng = np.random.default_rng(22)
    a = rng.standard_normal((M, K))
    b = rng.standard_normal((K, N))
    out = np.zeros((M, N))

    compiled = tilelang.compile(gemm, target=X100_TARGET, execution_backend="tvm_ffi")
    run_qemu(
        compiled,
        [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(out, False)],
        tmp_path / "gemm_blas_f64.so",
        vlen=X100_VLEN,
        extra_env={"OPENBLAS_NUM_THREADS": "1"},
    )
    np_assert_close(out, a @ b, rtol=1e-10, atol=1e-10)


@requires_qemu_stack
@requires_openblas
def test_gemm_blas_fused_with_tilelang(tmp_path, monkeypatch):
    """BLAS GEMM fused with TileLang epilogue code in one TIR kernel."""

    monkeypatch.setenv("TILELANG_RISCV_LDFLAGS", f"-L{OPENBLAS_LIBDIR} -lopenblas -lm")

    M, N, K = 32, 32, 32

    @T.prim_func
    def gemm_relu(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1):
            blas.gemm_blas(A, B, C, M, N, K)
            for i, j in T.grid(M, N):
                if C[i, j] < 0.0:
                    C[i, j] = 0.0

    rng = np.random.default_rng(23)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    out = np.zeros((M, N), dtype=np.float32)

    compiled = tilelang.compile(gemm_relu, target=X100_TARGET, execution_backend="tvm_ffi")
    run_qemu(
        compiled,
        [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(out, False)],
        tmp_path / "gemm_relu_blas.so",
        vlen=X100_VLEN,
        extra_env={"OPENBLAS_NUM_THREADS": "1"},
    )
    expected = np.maximum(a @ b, 0)
    np_assert_close(out, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    tilelang.testing.main()
