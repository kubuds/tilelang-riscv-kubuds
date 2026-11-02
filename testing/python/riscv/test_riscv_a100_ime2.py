"""Phase-3: K3 A100 support -- RVV VLEN=1024 and the SpacemiT IME-2.0 engine.

The A100 AI-CPU cluster combines

* an RVV 1.0 vector unit with ``VLEN=1024`` (32 architectural vector registers
  of 128B, see the SpacemiT K3 paper, section 3.2.1), and
* the shared IME-2.0 matrix engine (1024-bit MAC datapath, one 8x16x8 INT8 or
  8x8x8 FP16/BF16 MMA per instruction, operands/results in vector registers,
  section 3.2.2 and Table 3).

The IME-2.0 intrinsics (:mod:`tilelang.riscv.ime`) lower to the
``spacemit_ime2`` runtime ABI.  For correctness testing we link the reference
simulator ``src/riscv/ime2/ime2_sim.cc`` (built by the harness CMake project)
and execute under QEMU with ``VLEN=1024``.  On real silicon the vendor runtime
provides the same entry points.

Note: code compiled with the ``k3-a100`` preset may assume ``VLEN=1024``
(``+zvl1024b``) and must run on matching hardware (or QEMU with ``vlen=1024``).
"""

import numpy as np
import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.riscv import ime
from tilelang.riscv.qemu import QEMUTensor

from conftest import default_ime2_sim_dir, np_assert_close, requires_ime2_sim, requires_qemu_stack, run_qemu

A100_TARGET = "k3-a100"
A100_VLEN = 1024


def _tiled_matmul_i8_kernel(M, N, K):
    """Full i8 GEMM built from the 8x16x8 IME-2.0 micro-kernel."""

    @T.prim_func
    def gemm_i8(
        A: T.Tensor((M, K), "int8"),
        B: T.Tensor((K, N), "int8"),
        C: T.Tensor((M, N), "int32"),
    ):
        with T.Kernel(1):
            A_local = T.alloc_local((8, 16), "int8")
            B_local = T.alloc_local((16, 8), "int8")
            C_local = T.alloc_local((8, 8), "int32")
            for i0 in T.serial(M // 8):
                for j0 in T.serial(N // 8):
                    for i in T.serial(8):
                        for j in T.serial(8):
                            C_local[i, j] = 0
                    for k0 in T.serial(K // 16):
                        T.copy(A[i0 * 8, k0 * 16], A_local)
                        T.copy(B[k0 * 16, j0 * 8], B_local)
                        ime.ime2_mma_i8(A_local, B_local, C_local, C_local)
                    T.copy(C_local, C[i0 * 8, j0 * 8])

    return gemm_i8


def _tiled_matmul_f16_kernel(M, N, K, routine):
    """Full fp16/bf16 GEMM built from the 8x8x8 IME-2.0 micro-kernel."""

    @T.prim_func
    def gemm_f16(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1):
            A_local = T.alloc_local((8, 8), "float16")
            B_local = T.alloc_local((8, 8), "float16")
            C_local = T.alloc_local((8, 8), "float32")
            for i0 in T.serial(M // 8):
                for j0 in T.serial(N // 8):
                    for i in T.serial(8):
                        for j in T.serial(8):
                            C_local[i, j] = 0
                    for k0 in T.serial(K // 8):
                        T.copy(A[i0 * 8, k0 * 8], A_local)
                        T.copy(B[k0 * 8, j0 * 8], B_local)
                        routine(A_local, B_local, C_local, C_local)
                    T.copy(C_local, C[i0 * 8, j0 * 8])

    return gemm_f16


@requires_qemu_stack
def test_vector_add_f32_rvv1024(tmp_path):
    N = 1024

    @T.prim_func
    def vadd(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            for i in T.vectorized(N):
                C[i] = A[i] + B[i]

    rng = np.random.default_rng(30)
    a = rng.standard_normal(N).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)

    compiled = tilelang.compile(vadd, target=A100_TARGET, execution_backend="tvm_ffi")
    run_qemu(compiled, [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(out, False)], tmp_path / "vadd_rvv1024.so", vlen=A100_VLEN)
    np_assert_close(out, a + b, rtol=1e-6, atol=1e-6)


def test_uses_full_1024bit_vector_register():
    """The VLEN-aware vectorizer must widen f32 vectors to 32 lanes on A100.

    Checked on the LLVM IR fed to the code generator (the vector width is the
    vectorizer's direct output; disassembler-level checks for RVV presence live
    in test_riscv_rvv256).
    """

    from tilelang import tvm
    from tilelang.backend.target import determine_target
    from tilelang.engine import lower

    N = 1024

    @T.prim_func
    def vadd(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            for i in T.vectorized(N):
                C[i] = A[i] + B[i]

    target = determine_target(A100_TARGET, return_object=True)
    with tvm.target.Target(target):
        art = lower(vadd, target=target, enable_host_codegen=True, enable_device_compile=True)
    ir = art.rt_mod.inspect_source()
    assert "<32 x float>" in ir, "expected 32-lane (1024-bit) f32 vectors on k3-a100"
    assert "<4 x float>" not in ir, "128-bit vector width should not survive on k3-a100"


@requires_qemu_stack
@requires_ime2_sim
def test_ime2_mma_i8_microkernel(tmp_path, monkeypatch):
    """Single 8x16x8 IME-2.0 INT8 MMA against the reference simulator."""

    monkeypatch.setenv("TILELANG_RISCV_LDFLAGS", f"-L{default_ime2_sim_dir()} -lspacemit_ime2_sim")

    @T.prim_func
    def mma(
        A: T.Tensor((8, 16), "int8"),
        B: T.Tensor((16, 8), "int8"),
        C: T.Tensor((8, 8), "int32"),
        D: T.Tensor((8, 8), "int32"),
    ):
        with T.Kernel(1):
            ime.ime2_mma_i8(A, B, C, D)

    rng = np.random.default_rng(31)
    a = rng.integers(-8, 8, size=(8, 16)).astype(np.int8)
    b = rng.integers(-8, 8, size=(16, 8)).astype(np.int8)
    c = rng.integers(-100, 100, size=(8, 8)).astype(np.int32)
    out = np.zeros((8, 8), dtype=np.int32)

    compiled = tilelang.compile(mma, target=A100_TARGET, execution_backend="tvm_ffi")
    run_qemu(
        compiled,
        [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(c, True), QEMUTensor(out, False)],
        tmp_path / "ime2_i8.so",
        vlen=A100_VLEN,
    )
    expected = a.astype(np.int32) @ b.astype(np.int32) + c
    np_assert_close(out, expected, rtol=0, atol=0)


@requires_qemu_stack
@requires_ime2_sim
def test_ime2_mma_i8_tiled_gemm(tmp_path, monkeypatch):
    """Full i8 GEMM assembled from 8x16x8 IME-2.0 micro-kernels."""

    monkeypatch.setenv("TILELANG_RISCV_LDFLAGS", f"-L{default_ime2_sim_dir()} -lspacemit_ime2_sim")

    M, N, K = 32, 24, 48
    gemm_i8 = _tiled_matmul_i8_kernel(M, N, K)

    rng = np.random.default_rng(32)
    a = rng.integers(-4, 4, size=(M, K)).astype(np.int8)
    b = rng.integers(-4, 4, size=(K, N)).astype(np.int8)
    out = np.zeros((M, N), dtype=np.int32)

    compiled = tilelang.compile(gemm_i8, target=A100_TARGET, execution_backend="tvm_ffi")
    run_qemu(
        compiled,
        [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(out, False)],
        tmp_path / "ime2_gemm_i8.so",
        vlen=A100_VLEN,
    )
    expected = a.astype(np.int32) @ b.astype(np.int32)
    np_assert_close(out, expected, rtol=0, atol=0)


@requires_qemu_stack
@requires_ime2_sim
def test_ime2_mma_f16_microkernel(tmp_path, monkeypatch):
    """Single 8x8x8 IME-2.0 FP16 MMA."""

    monkeypatch.setenv("TILELANG_RISCV_LDFLAGS", f"-L{default_ime2_sim_dir()} -lspacemit_ime2_sim")

    @T.prim_func
    def mma(
        A: T.Tensor((8, 8), "float16"),
        B: T.Tensor((8, 8), "float16"),
        C: T.Tensor((8, 8), "float32"),
        D: T.Tensor((8, 8), "float32"),
    ):
        with T.Kernel(1):
            ime.ime2_mma_f16(A, B, C, D)

    rng = np.random.default_rng(33)
    a = rng.standard_normal((8, 8)).astype(np.float16)
    b = rng.standard_normal((8, 8)).astype(np.float16)
    c = rng.standard_normal((8, 8)).astype(np.float32)
    out = np.zeros((8, 8), dtype=np.float32)

    compiled = tilelang.compile(mma, target=A100_TARGET, execution_backend="tvm_ffi")
    run_qemu(
        compiled,
        [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(c, True), QEMUTensor(out, False)],
        tmp_path / "ime2_f16.so",
        vlen=A100_VLEN,
    )
    expected = a.astype(np.float32) @ b.astype(np.float32) + c
    np_assert_close(out, expected, rtol=1e-3, atol=1e-3)


@requires_qemu_stack
@requires_ime2_sim
def test_ime2_mma_f16_tiled_gemm(tmp_path, monkeypatch):
    """Full fp16 GEMM assembled from 8x8x8 IME-2.0 micro-kernels."""

    monkeypatch.setenv("TILELANG_RISCV_LDFLAGS", f"-L{default_ime2_sim_dir()} -lspacemit_ime2_sim")

    M, N, K = 16, 16, 24
    gemm_f16 = _tiled_matmul_f16_kernel(M, N, K, ime.ime2_mma_f16)

    rng = np.random.default_rng(34)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)
    out = np.zeros((M, N), dtype=np.float32)

    compiled = tilelang.compile(gemm_f16, target=A100_TARGET, execution_backend="tvm_ffi")
    run_qemu(
        compiled,
        [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(out, False)],
        tmp_path / "ime2_gemm_f16.so",
        vlen=A100_VLEN,
    )
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np_assert_close(out, expected, rtol=1e-2, atol=1e-2)


@requires_qemu_stack
@requires_ime2_sim
def test_ime2_mma_bf16_microkernel(tmp_path, monkeypatch):
    """Single 8x8x8 IME-2.0 BF16 MMA (BF16 operands via ml_dtypes)."""

    monkeypatch.setenv("TILELANG_RISCV_LDFLAGS", f"-L{default_ime2_sim_dir()} -lspacemit_ime2_sim")

    @T.prim_func
    def mma(
        A: T.Tensor((8, 8), "bfloat16"),
        B: T.Tensor((8, 8), "bfloat16"),
        C: T.Tensor((8, 8), "float32"),
        D: T.Tensor((8, 8), "float32"),
    ):
        with T.Kernel(1):
            ime.ime2_mma_bf16(A, B, C, D)

    import ml_dtypes

    rng = np.random.default_rng(35)
    a32 = rng.standard_normal((8, 8)).astype(np.float32)
    b32 = rng.standard_normal((8, 8)).astype(np.float32)
    a = a32.astype(ml_dtypes.bfloat16)
    b = b32.astype(ml_dtypes.bfloat16)
    c = rng.standard_normal((8, 8)).astype(np.float32)
    out = np.zeros((8, 8), dtype=np.float32)

    compiled = tilelang.compile(mma, target=A100_TARGET, execution_backend="tvm_ffi")
    run_qemu(
        compiled,
        [QEMUTensor(a, True), QEMUTensor(b, True), QEMUTensor(c, True), QEMUTensor(out, False)],
        tmp_path / "ime2_bf16.so",
        vlen=A100_VLEN,
    )
    expected = a.astype(np.float32) @ b.astype(np.float32) + c
    np_assert_close(out, expected, rtol=1e-2, atol=1e-2)


def test_ime2_intrinsic_shape_validation():
    with pytest.raises(ValueError, match="A must be"):

        @T.prim_func
        def bad(
            A: T.Tensor((4, 16), "int8"),
            B: T.Tensor((16, 8), "int8"),
            C: T.Tensor((8, 8), "int32"),
            D: T.Tensor((8, 8), "int32"),
        ):
            with T.Kernel(1):
                ime.ime2_mma_i8(A, B, C, D)


if __name__ == "__main__":
    tilelang.testing.main()
