"""SpacemiT IME-2.0 matrix-engine MMA intrinsics (K3 A100).

The K3 A100 AI-CPU provides the SpacemiT-IME matrix extensions.  IME-2.0 is
the second generation: a 1024-bit matrix multiply-accumulate datapath shared
by two cores (see the SpacemiT K3 paper, Table 3):

===============  =====================  ==================
mode             operands               accumulate/result
===============  =====================  ==================
IME-2.0 INT      A i8 (8x16), B (16x8)  C/D i32 (8x8)
IME-2.0 FP       A/B fp16|bf16 (8x8)    C/D fp32 (8x8)
===============  =====================  ==================

with ``D = A * B + C`` and operands/results held in vector registers.

Programming model
-----------------
Because the IME-2.0 instruction encodings are proprietary (they are not part
of the public RISC-V ISA specifications), TileLang exposes the engine through
a stable *runtime ABI*: the TIR intrinsics below lower to extern calls of the
``spacemit_ime2`` runtime library::

    void spacemit_ime2_mma_i8(const int8_t* A, const int8_t* B,
                              const int32_t* C, int32_t* D);
    void spacemit_ime2_mma_f16(const void* A, const void* B,
                               const float* C, float* D);
    void spacemit_ime2_mma_bf16(const void* A, const void* B,
                                const float* C, float* D);

Two implementations of that ABI exist:

* ``src/riscv/ime2/ime2_sim.cc`` -- a reference simulator (plain RV64GC C)
  used to validate kernel correctness under QEMU and on any host; it is the
  executable specification of the semantics,
* the vendor runtime on real silicon, which issues the actual IME-2.0
  instructions (and may use register-resident fast paths internally).

The boundary is deliberately at the extern-call level so kernels stay in
TileLang/TIR and remain portable across the simulator and the hardware
runtime.

Usage example (full i8 GEMM built from 8x16x8 micro-kernels)::

    @T.prim_func
    def gemm_i8(A: T.Tensor((M, K), "int8"), B: T.Tensor((K, N), "int8"),
                C: T.Tensor((M, N), "int32")):
        with T.Kernel(1):
            for i0 in T.serial(M // 8):
                for j0 in T.serial(N // 8):
                    for k0 in T.serial(K // 16):
                        ime.ime2_mma_i8(
                            A[i0 * 8:i0 * 8 + 8, k0 * 16:k0 * 16 + 16],
                            B[k0 * 16:k0 * 16 + 16, j0 * 8:j0 * 8 + 8],
                            C[i0 * 8:i0 * 8 + 8, j0 * 8:j0 * 8 + 8],
                            C[i0 * 8:i0 * 8 + 8, j0 * 8:j0 * 8 + 8])

License: MIT (original code written for TileLang; the operator shapes and
data types are facts from the SpacemiT K3 paper).
"""

from __future__ import annotations

from typing import Any

import tilelang.language as T

IME2_INT_SHAPE = (8, 16, 8)  # M, K, N of the INT8 micro-kernel
IME2_FP_SHAPE = (8, 8, 8)  # M, K, N of the FP16/BF16 micro-kernel


def _shape_of(buf) -> tuple[int, ...]:
    shape = getattr(buf, "shape", None)
    if shape is None:
        raise TypeError(f"expected a buffer with a shape, got {type(buf)}")
    return tuple(int(dim) for dim in shape)


def ime2_mma_i8(A, B, C, D) -> Any:
    """Emit an IME-2.0 INT8 MMA: ``D(8x8 i32) = A(8x16 i8) * B(16x8 i8) + C(8x8 i32)``.

    ``D`` may alias ``C`` (in-place accumulate).  The expression lowers to the
    ``spacemit_ime2_mma_i8`` runtime call.
    """
    if _shape_of(A) != (8, 16):
        raise ValueError(f"A must be (8, 16), got {_shape_of(A)}")
    if _shape_of(B) != (16, 8):
        raise ValueError(f"B must be (16, 8), got {_shape_of(B)}")
    if _shape_of(C) != (8, 8) or _shape_of(D) != (8, 8):
        raise ValueError(f"C and D must be (8, 8), got {_shape_of(C)}, {_shape_of(D)}")
    return T.call_extern(
        "handle",
        "spacemit_ime2_mma_i8",
        T.access_ptr(A, "r"),
        T.access_ptr(B, "r"),
        T.access_ptr(C, "r"),
        T.access_ptr(D, "w"),
    )


def _ime2_mma_f(A, B, C, D, routine: str) -> Any:
    if _shape_of(A) != (8, 8) or _shape_of(B) != (8, 8):
        raise ValueError(f"A and B must be (8, 8), got {_shape_of(A)}, {_shape_of(B)}")
    if _shape_of(C) != (8, 8) or _shape_of(D) != (8, 8):
        raise ValueError(f"C and D must be (8, 8), got {_shape_of(C)}, {_shape_of(D)}")
    return T.call_extern(
        "handle",
        routine,
        T.access_ptr(A, "r"),
        T.access_ptr(B, "r"),
        T.access_ptr(C, "r"),
        T.access_ptr(D, "w"),
    )


def ime2_mma_f16(A, B, C, D) -> Any:
    """Emit an IME-2.0 FP16 MMA: ``D(8x8 f32) = A(8x8 f16) * B(8x8 f16) + C(8x8 f32)``.

    Lowers to the ``spacemit_ime2_mma_f16`` runtime call.
    """
    return _ime2_mma_f(A, B, C, D, "spacemit_ime2_mma_f16")


def ime2_mma_bf16(A, B, C, D) -> Any:
    """Emit an IME-2.0 BF16 MMA: ``D(8x8 f32) = A(8x8 bf16) * B(8x8 bf16) + C(8x8 f32)``.

    Lowers to the ``spacemit_ime2_mma_bf16`` runtime call.
    """
    return _ime2_mma_f(A, B, C, D, "spacemit_ime2_mma_bf16")
