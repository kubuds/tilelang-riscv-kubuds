"""IME2 (smt.vmadot) int8 GEMM and RVV-at-VLEN=1024 on the K3 A100 cluster.

Run:  python testing/riscv/test_riscv_ime2.py
      (needs libtl_riscv_ime2.so in build/lib and write access to
       /proc/set_ai_thread, which is world-writable on Bianbu)

The process is moved to cpu8-15 before any kernel is built: IME2 instructions
only exist there, and a normal process (affinity 0-7) gets SIGILL.  This is
process-wide and cannot be undone, which is why it lives in a test process and
not in the library.

Packing is timed separately from the GEMM on purpose.  Folding pack_a/pack_b
into the timed call measures the packer (which is scalar and runs on the slow
A100 scalar units), not the matrix unit: with packing included the same 1024^3
problem reads 38 GMAC/s, the bare kernel reads ~1000.
"""

from __future__ import annotations

import os
import time

import numpy as np
import tilelang
import tilelang.language as T
from tilelang import tvm


def claim_ai_cluster() -> str:
    with open("/proc/set_ai_thread", "w") as f:
        f.write(f"{os.getpid()}\n")
    return f"affinity {sorted(os.sched_getaffinity(0))}"


def build_pack(M, N, K):
    @T.prim_func
    def kern(A: T.Tensor((M * K,), "uint8"), B: T.Tensor((K * N,), "uint8"),
             Ap: T.Tensor((M * K,), "uint8"), Bp: T.Tensor((N * K,), "uint8")):
        with T.Kernel(1):
            T.call_extern("int32", "tl_riscv_ime2_pack_b", N, K,
                          T.address_of(B[0]), T.address_of(Bp[0]))
            T.call_extern("int32", "tl_riscv_ime2_pack_a", M, K,
                          T.address_of(A[0]), T.address_of(Ap[0]))

    return kern


def build_gemm(M, N, K):
    @T.prim_func
    def kern(Ap: T.Tensor((M * K,), "uint8"), Bp: T.Tensor((N * K,), "uint8"),
             C: T.Tensor((M * N,), "int32")):
        with T.Kernel(1):
            T.call_extern("int32", "tl_riscv_ime2_gemm", M, N, K,
                          T.address_of(Ap[0]), T.address_of(Bp[0]),
                          T.address_of(C[0]), N)

    return kern


def build_gemv(N, K):
    @T.prim_func
    def kern(W: T.Tensor((N * K,), "uint8"),
             S: T.Tensor((N * K // 32 * 2,), "uint8"),
             X: T.Tensor((K,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            T.call_extern("int32", "tl_riscv_gemv_q8", N, K,
                          T.address_of(W[0]), T.address_of(S[0]),
                          T.address_of(X[0]), T.address_of(Y[0]))

    return kern


def main() -> int:
    print(claim_ai_cluster(), flush=True)
    rng = np.random.default_rng(0)
    ok = True

    ime2_cfg = {"kind": "riscv", "march": "rv64gcv_zvl1024b", "mabi": "lp64d",
                "vlen": 1024, "ime2": True}
    for M, N, K in ((64, 64, 64), (256, 256, 256), (512, 512, 512),
                    (1024, 1024, 1024)):
        a = rng.integers(-127, 127, (M, K), dtype=np.int8)
        b = rng.integers(-127, 127, (K, N), dtype=np.int8)
        ref = a.astype(np.int32) @ b.astype(np.int32)
        with tvm.target.Target(ime2_cfg):
            fp = tilelang.compile(build_pack(M, N, K), out_idx=None,
                                  execution_backend="tvm_ffi")
            fg = tilelang.compile(build_gemm(M, N, K), out_idx=None,
                                  execution_backend="tvm_ffi")
        A = np.zeros(M * K, np.uint8)
        B = np.zeros(K * N, np.uint8)
        Ap = np.zeros(M * K, np.uint8)
        Bp = np.zeros(N * K, np.uint8)
        C = np.zeros(M * N, np.int32)
        A.view(np.int8)[:] = a.reshape(-1)
        B.view(np.int8)[:] = b.reshape(-1)

        t0 = time.time()
        fp(A, B, Ap, Bp)
        pack_s = time.time() - t0
        fg(Ap, Bp, C)
        err = int(np.abs(C.reshape(M, N) - ref).max())
        ok &= err == 0

        fg(Ap, Bp, C)
        best = 1e9
        for _ in range(5):
            t = time.time()
            fg(Ap, Bp, C)
            best = min(best, time.time() - t)
        gmac = M * N * K / best / 1e9
        print(f"  ime2 {M}x{N}x{K}: max|err|={err}  gemm {best*1e3:6.2f} ms  "
              f"{gmac:7.1f} GMAC/s  (pack once {pack_s*1e3:.1f} ms)",
              flush=True)

    # RVV on the A100 cluster (VLEN=1024): correctness only.  Generic RVV code
    # is ~25x slower on A100 than on X100 by design (README of this backend).
    N, K = 256, 1024
    cfg = {"kind": "riscv", "march": "rv64gcv_zvl1024b", "mabi": "lp64d",
           "vlen": 1024}
    w = rng.standard_normal((N, K)).astype(np.float32) * 0.02
    sc = np.abs(w.reshape(N, K // 32, 32)).max(axis=2) / 127.0
    q = np.round(w.reshape(N, K // 32, 32) / sc[:, :, None]).clip(-127, 127).astype(np.int8)
    x = rng.standard_normal(K).astype(np.float32)
    with tvm.target.Target(cfg):
        g = tilelang.compile(build_gemv(N, K), out_idx=None,
                             execution_backend="tvm_ffi")
    y = np.zeros(N, np.float32)
    g(q.reshape(-1).view(np.uint8), sc.astype(np.float16).view(np.uint8).reshape(-1), x, y)
    ref = (q.reshape(N, K).astype(np.float32) * sc.repeat(32, axis=1)) @ x
    err = float(np.abs(y - ref).max())
    ok &= err < 2e-3
    print(f"  rvv vlen=1024 gemv {N}x{K}: max|err|={err:.2e}", flush=True)

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
