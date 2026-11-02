"""TIR-level GEMM offload to a BLAS library (OpenBLAS) for RISC-V CPUs.

Motivation
----------
The SpacemiT K3 X100 (RVV 1.0, VLEN=256) benefits from hand-tuned BLAS
micro-kernels: OpenBLAS ships RVV GEMM kernels for ``RISCV64_ZVL256B``
(e.g. ``sgemm_kernel_16x8_zvl256b.c``) that are hard to beat with generic
vectorization at this stage.  TileLang kernels can offload GEMM work to such a
library entirely within TIR: the operator lowers to a ``tir.call_extern`` of
the CBLAS API, the exported kernel shared library links against the BLAS
library, and everything else (fusion with surrounding TileLang code, buffers,
scheduling) stays in the TileLang/TIR route.

Building OpenBLAS for the X100 (from the OpenBLAS source tree)::

    make -j CC="riscv64-linux-gnu-gcc -fPIC" HOSTCC=gcc \\
         TARGET=RISCV64_ZVL256B NOFORTRAN=1 USE_THREAD=0 NUM_THREADS=1 \\
         NO_SHARED=1 BINARY=64

and point the export at it with ``TILELANG_RISCV_LDFLAGS`` (see
:mod:`tilelang.riscv.export`), e.g.::

    export TILELANG_RISCV_LDFLAGS="-L/path/to/OpenBLAS -lopenblas -lm"

License note: OpenBLAS is BSD-3-Clause licensed and *linked*, never copied
into TileLang; the license is compatible with TileLang's MIT license.

Caveats
-------
* Only ``float32``/``float64`` GEMM is exposed (the standard CBLAS GEMM
  entry points ``cblas_sgemm``/``cblas_dgemm``).
* The kernel shared library must resolve ``cblas_*`` at load time: link the
  BLAS library into the exported library (static linking is recommended for
  deployment simplicity).
"""

from __future__ import annotations

from typing import Any

import tilelang.language as T

# CBLAS enums (cblas.h).
CBLAS_ROW_MAJOR = 101
CBLAS_COL_MAJOR = 102
CBLAS_NO_TRANS = 111
CBLAS_TRANS = 112

_GEMM_ROUTINES = {
    "float32": "cblas_sgemm",
    "float64": "cblas_dgemm",
}


def _shape_of(buf) -> tuple[int, ...]:
    shape = getattr(buf, "shape", None)
    if shape is None:
        raise TypeError(f"expected a buffer with a shape, got {type(buf)}")
    return tuple(int(dim) for dim in shape)


def gemm_blas(
    A,
    B,
    C,
    M: int,
    N: int,
    K: int,
    *,
    dtype: str = "float32",
    transpose_a: bool = False,
    transpose_b: bool = False,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Any:
    """Emit a row-major GEMM ``D = alpha * op(A) * op(B) + beta * C`` as a TIR extern call.

    ``C`` holds both the accumulator input (scaled by ``beta``) and the output,
    matching the CBLAS contract.  With ``beta=0.0`` the ``C`` input is ignored
    by the BLAS kernel (no read is performed), so no initialization of ``C`` is
    required for the common ``C = A @ B`` case.

    Shapes: ``A`` is ``(M, K)`` (or ``(K, M)`` with ``transpose_a``), ``B`` is
    ``(K, N)`` (or ``(N, K)`` with ``transpose_b``), ``C`` is ``(M, N)``.
    """
    if dtype not in _GEMM_ROUTINES:
        raise ValueError(f"gemm_blas supports {sorted(_GEMM_ROUTINES)}, got {dtype!r}")
    routine = _GEMM_ROUTINES[dtype]

    a_shape = _shape_of(A)
    b_shape = _shape_of(B)
    c_shape = _shape_of(C)
    expected_a = (K, M) if transpose_a else (M, K)
    expected_b = (N, K) if transpose_b else (K, N)
    if c_shape != (M, N):
        raise ValueError(f"C shape {c_shape} != {(M, N)}")
    if a_shape != expected_a:
        raise ValueError(f"A shape {a_shape} != {expected_a} (transpose_a={transpose_a})")
    if b_shape != expected_b:
        raise ValueError(f"B shape {b_shape} != {expected_b} (transpose_b={transpose_b})")

    # Row-major leading dimensions equal the number of columns of the stored
    # (untransposed) matrices.
    lda = a_shape[1]
    ldb = b_shape[1]
    ldc = c_shape[1]

    scalar_ctor = T.float32 if dtype == "float32" else T.float64
    return T.call_extern(
        "handle",
        routine,
        CBLAS_ROW_MAJOR,
        CBLAS_TRANS if transpose_a else CBLAS_NO_TRANS,
        CBLAS_TRANS if transpose_b else CBLAS_NO_TRANS,
        M,
        N,
        K,
        scalar_ctor(alpha),
        T.access_ptr(A, "r"),
        lda,
        T.access_ptr(B, "r"),
        ldb,
        scalar_ctor(beta),
        T.access_ptr(C, "w"),
        ldc,
    )
