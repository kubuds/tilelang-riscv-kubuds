#!/usr/bin/env python3
"""Step 2 probe: does the riscv backend actually reach OpenBLAS?

Two kernels, same math (512^3 fp32 GEMM):
  A) the current TIR loop nest        -- scalar/auto-vectorised, ~226 ms
  B) a T.call_extern into cblas_sgemm -- the route T.gemm lowering will take

B is not yet a general lowering: it is a hand-written extern call used to
prove three things before investing in the lowering pass:
  1. `blas=openblas` links the system OpenBLAS on this board,
  2. cblas.h is visible to the generated C (via -include),
  3. the resulting speed is what the hardware can do.

Row-major is passed directly: OpenBLAS supports CblasRowMajor, so no
transpose trick is needed.  Enum values are from cblas.h (101 RowMajor,
111 NoTrans) and are spelled out here because the DSL cannot import C enums.
"""

from __future__ import annotations

import json
import sys
import time

import torch
import tilelang
import tilelang.cpu.language as T
from tilelang import tvm

M = N = K = 512


def kernel_loops():
    @T.prim_func
    def matmul(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1) as bx:
            A_l = T.alloc_local((M, K), "float32")
            B_l = T.alloc_local((K, N), "float32")
            C_l = T.alloc_local((M, N), "float32")
            for i in T.serial(M):
                for j in T.serial(N):
                    C_l[i, j] = 0
            for i in T.serial(M):
                for kk in T.serial(K):
                    A_l[i, kk] = A[i, kk]
            for kk in T.serial(K):
                for j in T.serial(N):
                    B_l[kk, j] = B[kk, j]
            for i in T.serial(M):
                for j in T.serial(N):
                    for kk in T.serial(K):
                        C_l[i, j] += A_l[i, kk] * B_l[kk, j]
            for i in T.serial(M):
                for j in T.serial(N):
                    C[i, j] = C_l[i, j]

    return matmul


def kernel_blas():
    @T.prim_func
    def matmul_blas(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1) as bx:
            T.call_extern(
                "int32",
                "cblas_sgemm",
                101,  # CblasRowMajor
                111,  # CblasNoTrans
                111,  # CblasNoTrans
                M,
                N,
                K,
                1.0,  # alpha
                T.address_of(A[0, 0]),
                K,  # lda
                T.address_of(B[0, 0]),
                N,  # ldb
                0.0,  # beta
                T.address_of(C[0, 0]),
                N,  # ldc
            )

    return matmul_blas


def run(name, fn, cfg):
    t0 = time.time()
    with tvm.target.Target(cfg):
        f = tilelang.compile(fn(), -1, execution_backend="tvm_ffi")
    build = time.time() - t0

    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)
    ref = a @ b

    best = float("inf")
    for _ in range(3):
        t0 = time.time()
        c = f(a, b)
        best = min(best, time.time() - t0)

    err = (c - ref).abs().max().item()
    gf = 2 * M * N * K / best / 1e9
    print(f"   {name:10s} build {build:5.1f}s  {best*1e3:8.1f} ms  {gf:6.2f} GFLOP/s  max abs {err:.3e}")
    return best, err


def main() -> int:
    cfg = {
        "kind": "riscv",
        "march": "rv64gcv",
        "mabi": "lp64d",
        "vlen": 256,
        "blas": "openblas",
    }
    if len(sys.argv) > 1:
        cfg = json.loads(sys.argv[1])
    print(f"== target: {cfg}")
    run("loops", kernel_loops, cfg)
    run("cblas", kernel_blas, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
