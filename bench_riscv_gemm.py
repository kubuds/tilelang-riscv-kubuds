"""fp32 GEMM on the riscv target: scalar vs in-tree RVV vs OpenBLAS.

All three go through the same T.gemm kernel; only the target attributes
change, so the numbers isolate the GEMM lowering, not the schedule.
"""

import sys
import time

import torch
import tilelang
import tilelang.language as T
from tilelang import tvm


def kernel(M, N, K, BM, BN, BK):
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


def run(label, cfg, M, N, K, BM, BN, BK):
    t0 = time.time()
    with tvm.target.Target(cfg):
        f = tilelang.compile(kernel(M, N, K, BM, BN, BK), -1, execution_backend="tvm_ffi")
    build = time.time() - t0

    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)
    ref = a @ b
    f(a, b)  # warm up
    best = 1e9
    for _ in range(3):
        t = time.time()
        c = f(a, b)
        best = min(best, time.time() - t)
    err = (c - ref).abs().max().item()
    print(
        f"   {label:26s} build {build:5.1f}s  {best * 1e3:8.1f} ms  "
        f"{2 * M * N * K / best / 1e9:6.2f} GFLOP/s  max abs {err:.2e}",
        flush=True,
    )
    return 2 * M * N * K / best / 1e9


def main() -> int:
    M = N = K = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    BM = BN = BK = 128
    base = {"kind": "riscv", "march": "rv64gcv", "mabi": "lp64d", "vlen": 256}
    print(f"{M}^3 fp32, tiles {BM}x{BN}x{BK}")
    run("scalar", dict(base), M, N, K, BM, BN, BK)
    run("rvv template", {**base, "rvv": True}, M, N, K, BM, BN, BK)
    run("openblas", {**base, "blas": "openblas"}, M, N, K, BM, BN, BK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
