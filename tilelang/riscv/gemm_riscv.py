"""GEMM implementation for the RISC-V target.

Why this file has to exist at all
---------------------------------
Upstream v0.1.15 has no GEMM implementation for the CPU family: C++ SelectInst
returns the key "cpu.scalar" for anything kDLCPU, but nothing registers it, so
`T.gemm` fails on target "c" (layout cannot be inferred) and on "riscv"
("No GEMM implementation registered").  This module supplies the first one.

Two lowerings, one entry point
------------------------------
* blas == "openblas": emit a single cblas_sgemm call over the tile.  Measured
  on K3 X100: 51.75 GFLOP/s for 512^3 fp32 vs 1.16 for the scalar loop.
* otherwise: the scalar FMA nest, modelled on
  tilelang/cuda/op/gemm/gemm_fma.py (same repository, Apache-2.0), reduced to
  the case that matters here: A and B are shared tiles, C is a fragment.

The FMA body is deliberately copied in shape rather than imported: gemm_fma
lives under tilelang/cuda and its staging/branching is written for warp-level
threads.  Keeping a small CPU-shaped copy avoids coupling this backend to the
CUDA package.
"""

from __future__ import annotations

from dataclasses import dataclass

from tvm import tirx
from tvm.ir import Range
from tvm.target import Target

from tilelang import language as T
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.tileop.gemm.registry import register_gemm_impl
from tilelang.utils.language import is_fragment


def _riscv_linear_fragment(buffer, thread_nums: int):
    """Row-major flattened fragment layout (one thread owns whole rows).

    Mirrors the linear layout used by the CUDA FMA fallback: the fragment is
    laid out so that element (i, j) lives at flat index i*N + j, making the
    scalar accumulate loop trivially addressable.
    """
    from tilelang.layout import make_linear_layout

    return make_linear_layout(buffer)


@dataclass
class RiscvGemm(GemmBase):
    """Scalar / BLAS GEMM for RISC-V CPU-class targets."""

    def infer_layout(self, target: Target, thread_nums: int):
        layouts = {}
        for buf in (self.A, self.B, self.C):
            if is_fragment(buf):
                layouts[buf] = _riscv_linear_fragment(buf, thread_nums)
        return layouts

    def _blas_enabled(self, target: Target) -> bool:
        return str(target.attrs.get("blas", "")) == "openblas"

    def _rvv_enabled(self, target: Target) -> bool:
        """True when the caller asked for the in-tree RVV GEMM.

        Kept separate from ``blas``: OpenBLAS wins on throughput, the RVV
        template is the fallback that needs no external library and is the
        only fp32 GEMM available on a bare rv64gcv userland.
        """
        return bool(target.attrs.get("rvv", False))

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        M, N, K = self.M, self.N, self.K
        A_r, B_r, C_r = self.ARegion, self.BRegion, self.CRegion
        A_buf, B_buf, C_buf = A_r.buffer, B_r.buffer, C_r.buffer
        a0, a1 = A_r.region[-2].min, A_r.region[-1].min
        b0, b1 = B_r.region[-2].min, B_r.region[-1].min
        c0, c1 = C_r.region[-2].min, C_r.region[-1].min
        a_cols = A_r.region[-1].extent
        b_cols = B_r.region[-1].extent
        c_cols = C_r.region[-1].extent
        accum_dtype = self.accum_dtype
        a_dtype, b_dtype = self.a_dtype, self.b_dtype
        thread_nums = thread_bounds.extent
        clear_accum = self.clear_accum

        if self._blas_enabled(target):

            @T.prim_func
            def _gemm_blas() -> None:
                T.call_extern(
                    "int32",
                    "cblas_sgemm",
                    101,  # CblasRowMajor
                    111,  # CblasNoTrans
                    111,  # CblasNoTrans
                    M,
                    N,
                    K,
                    1.0 if clear_accum else 1.0,
                    T.address_of(A_buf[tuple(r.min for r in A_r.region[:-2]) + (a0, a1)]),
                    a_cols,
                    T.address_of(B_buf[tuple(r.min for r in B_r.region[:-2]) + (b0, b1)]),
                    b_cols,
                    0.0 if clear_accum else 1.0,
                    T.address_of(C_buf[tuple(r.min for r in C_r.region[:-2]) + (c0, c1)]),
                    c_cols,
                )

            return _gemm_blas

        if self._rvv_enabled(target):

            @T.prim_func
            def _gemm_rvv() -> None:
                T.call_extern(
                    "int32",
                    "tl_riscv_gemm_f32",
                    M,
                    N,
                    K,
                    T.address_of(A_buf[tuple(r.min for r in A_r.region[:-2]) + (a0, a1)]),
                    a_cols,
                    T.address_of(B_buf[tuple(r.min for r in B_r.region[:-2]) + (b0, b1)]),
                    b_cols,
                    T.address_of(C_buf[tuple(r.min for r in C_r.region[:-2]) + (c0, c1)]),
                    c_cols,
                    0 if clear_accum else 1,
                )

            return _gemm_rvv

        @T.prim_func
        def _gemm_scalar() -> None:
            accum = T.alloc_local((1,), accum_dtype)
            for c_flat in T.serial(thread_index, M * N, thread_nums):
                i = c_flat // N
                j = c_flat % N
                if clear_accum:
                    accum[0] = T.cast(0, accum_dtype)
                else:
                    accum[0] = T.cast(C_buf[tuple(r.min for r in C_r.region[:-2]) + (c0 + i, c1 + j)], accum_dtype)
                for k in T.serial(K):
                    accum[0] += T.cast(
                        A_buf[tuple(r.min for r in A_r.region[:-2]) + (a0 + i, a1 + k)], accum_dtype
                    ) * T.cast(
                        B_buf[tuple(r.min for r in B_r.region[:-2]) + (b0 + k, b1 + j)], accum_dtype
                    )
                C_buf[tuple(r.min for r in C_r.region[:-2]) + (c0 + i, c1 + j)] = accum[0]

        return _gemm_scalar


def _matches_riscv(target: Target) -> bool:
    return target.kind.name == "riscv"


# inst_name must be "cpu.scalar": that is the key C++ SelectInst returns for
# every kDLCPU target, so it cannot be chosen freely here.
register_gemm_impl("riscv.gemm", "cpu.scalar", _matches_riscv, RiscvGemm)
