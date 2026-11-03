"""Throughput of T.gemm on the riscv target: scalar vs BLAS lowering."""

import json
import sys
import time

import torch
import tilelang
import tilelang.language as T
from tilelang import tvm

M = N = K = 512
BM = BN = BK = 128


def kernel():
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM)) as (bx, by):
            A_s = T.alloc_shared((BM, BK), "float32")
            B_s = T.alloc_shared((BK, BN), "float32")
            C_l = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_l)
            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=1):
                T.copy(A[by * BM, ko * BK], A_s)
                T.copy(B[ko * BK, bx * BN], B_s)
                T.gemm(A_s, B_s, C_l)
            T.copy(C_l, C[by * BM, bx * BN])

    return gemm


def run(label, cfg):
    t0 = time.time()
    with tvm.target.Target(cfg):
        f = tilelang.compile(kernel(), -1, execution_backend="tvm_ffi")
    build = time.time() - t0

    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)
    ref = a @ b
    best = min(
        (lambda: (lambda s: (f(a, b), time.time() - s)[1])(time.time()))() for _ in range(3)
    )
    c = f(a, b)
    err = (c - ref).abs().max().item()
    print(
        f"   {label:26s} build {build:5.1f}s  {best*1e3:8.1f} ms  "
        f"{2*M*N*K/best/1e9:6.2f} GFLOP/s  max abs {err:.2e}",
        flush=True,
    )


def main() -> int:
    base = {"kind": "riscv", "march": "rv64gcv", "mabi": "lp64d", "vlen": 256}
    run("rv64gcv scalar", dict(base))
    run("rv64gcv + openblas", {**base, "blas": "openblas"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
