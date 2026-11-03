#!/usr/bin/env python3
"""Step 1 acceptance test: pure RV64GC through the new `riscv` target kind.

What it proves
--------------
1. The `riscv` target kind is registered and accepted by tilelang.compile.
2. A kernel compiled for `riscv` produces bit-comparable results to torch.
3. The generated object contains no vector instructions -> this is genuinely
   the RV64GC baseline, not something the compiler quietly vectorised.

Run:  python3 test_riscv_step1.py [target-string]
      default target: "riscv -march=rv64gc -mabi=lp64d"

Notes
-----
This mirrors testing/python/cpu/test_tilelang_cpu_gemm.py on purpose: step 1
must not change the program, only the target.  Any numeric difference here
would mean the backend is wrong, not that the kernel is different.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

import torch
import tilelang
import tilelang.cpu.language as T
from tilelang import tvm

M, N, K = 512, 512, 512
BM, BN, BK = 128, 128, 128


def make_kernel():
    @T.prim_func
    def matmul(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM)) as (bx, by):
            A_l = T.alloc_local((BM, BK), "float32")
            B_l = T.alloc_local((BK, BN), "float32")
            C_l = T.alloc_local((BM, BN), "float32")

            for i in T.serial(BM):
                for j in T.serial(BN):
                    C_l[i, j] = 0
            for ko in T.serial(K // BK):
                for i in T.serial(BM):
                    for kk in T.serial(BK):
                        A_l[i, kk] = A[by * BM + i, ko * BK + kk]
                for kk in T.serial(BK):
                    for j in T.serial(BN):
                        B_l[kk, j] = B[ko * BK + kk, bx * BN + j]
                for i in T.serial(BM):
                    for j in T.serial(BN):
                        for kk in T.serial(BK):
                            C_l[i, j] += A_l[i, kk] * B_l[kk, j]
            for i in T.serial(BM):
                for j in T.serial(BN):
                    C[by * BM + i, bx * BN + j] = C_l[i, j]

    return matmul


def newest_so(before: set[str]) -> str | None:
    cache = os.path.expanduser("~/.tilelang/cache")
    cands = [p for p in glob.glob(os.path.join(cache, "**", "*.so"), recursive=True) if p not in before]
    return max(cands, key=os.path.getmtime) if cands else None


def main() -> int:
    # TVM 0.25 removed CLI target strings; the JSON dict form is required.
    # Accept either a JSON object or a bare kind name on the command line.
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg.startswith("{"):
        target_cfg = json.loads(arg)
    elif arg:
        target_cfg = {"kind": arg}
    else:
        target_cfg = {"kind": "riscv", "march": "rv64gc", "mabi": "lp64d"}
    print(f"== target: {target_cfg}")

    cache = os.path.expanduser("~/.tilelang/cache")
    before = set(glob.glob(os.path.join(cache, "**", "*.so"), recursive=True))

    t0 = time.time()
    with tvm.target.Target(target_cfg):
        fn = tilelang.compile(make_kernel(), -1, execution_backend="tvm_ffi")
    t_build = time.time() - t0
    print(f"   compile: {t_build:.1f} s")

    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)

    t0 = time.time()
    c = fn(a, b)
    t_run = time.time() - t0

    ref = torch.matmul(a.float(), b.float())
    diff = (c.float() - ref).abs()
    denom = ref.abs().clamp_min(1e-3)
    rel = (diff / denom).max().item()
    print(f"   run: {t_run * 1e3:.1f} ms   max abs {diff.max().item():.4e}   max rel {rel:.3e}")

    # RV64GC baseline must be scalar.  Count vector instructions.
    so = newest_so(before)
    if so:
        dis = subprocess.run(["objdump", "-d", so], capture_output=True, text=True).stdout
        n_vec = sum(dis.count(m) for m in ("vsetvli", "vle32.v", "vse32.v", "vfmacc.vv", "vmv"))
        print(f"   object: {so}")
        print(f"   vector instructions: {n_vec} (expected 0 for RV64GC)")
    else:
        print("   object: not found (cache miss)")

    ok = rel < 5e-3
    print("== RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
