"""Does the real T.gemm tile primitive work on the riscv target at all?

Every earlier test hand-wrote the triple loop, so the T.gemm lowering path has
never been exercised here.  If this fails, the "make T.gemm call BLAS" plan is
premature: there would be no working T.gemm to begin with.
"""

import json
import sys
import time

import torch
import tilelang
import tilelang.language as T
from tilelang import tvm

M = N = K = 128
BM = BN = BK = 64


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


def main() -> int:
    cfg = {"kind": "riscv", "march": "rv64gc", "mabi": "lp64d"}
    if len(sys.argv) > 1:
        cfg = json.loads(sys.argv[1])
    print(f"== target: {cfg}", flush=True)

    try:
        t0 = time.time()
        with tvm.target.Target(cfg):
            f = tilelang.compile(kernel(), -1, execution_backend="tvm_ffi")
        print(f"   compile ok: {time.time() - t0:.1f} s", flush=True)
    except Exception as exc:  # noqa: BLE001 - the failure mode is the result
        print(f"   COMPILE FAILED: {type(exc).__name__}: {str(exc)[:400]}", flush=True)
        return 1

    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)
    c = f(a, b)
    ref = a @ b
    err = (c - ref).abs().max().item()
    print(f"   run ok   max abs {err:.3e}", flush=True)
    print("== RESULT:", "PASS" if err < 1e-2 else "WRONG")
    return 0 if err < 1e-2 else 2


if __name__ == "__main__":
    raise SystemExit(main())
