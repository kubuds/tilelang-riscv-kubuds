"""Ceiling measurement: how fast could decode be if every GEMM ran on IME2?

Packs a Qwen3-0.6B layer's matrices once, then times the bare kernel with M=16
(the decode batch the prior two-cluster engine used) and projects a per-token
time.  This decides whether the two-process IME2 path is worth building at all.
"""

import ctypes
import os
import time

import numpy as np

LIB = ctypes.CDLL("/work/kubuds/hermes/tilelang_riscv/build/lib/libtl_riscv_ime2.so")
LIB.ime2_pack_b_only.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
LIB.ime2_pack_a_only.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
LIB.ime2_gemm_ldc.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2 + [ctypes.c_void_p, ctypes.c_int]


def claim_ai():
    with open("/proc/set_ai_thread", "w") as f:
        f.write(f"{os.getpid()}\n")
    return sorted(os.sched_getaffinity(0))


H = 1024
NQ = 2048
KV = 1024
IM = 3072
V = 151936
M = 16
L = 28


def bench(name, N, K, reps=5):
    rng = np.random.default_rng(0)
    B = rng.integers(-127, 127, (K, N), dtype=np.int8)          # K-major
    Bp = np.zeros((N // 32) * (K // 16) * 512, np.uint8)
    t0 = time.time()
    LIB.ime2_pack_b_only(N, K, B.ctypes.data, Bp.ctypes.data)
    pack_s = time.time() - t0
    A = np.zeros((M, K), np.int8)
    Ap = np.zeros((M // 16) * (K // 16) * 256, np.uint8)
    LIB.ime2_pack_a_only(M, K, A.ctypes.data, Ap.ctypes.data)
    C = np.zeros((M, N), np.int32)
    LIB.ime2_gemm_ldc(M, N, K, Ap.ctypes.data, Bp.ctypes.data, C.ctypes.data, N)
    best = 1e9
    for _ in range(reps):
        t = time.time()
        LIB.ime2_gemm_ldc(M, N, K, Ap.ctypes.data, Bp.ctypes.data, C.ctypes.data, N)
        best = min(best, time.time() - t)
    gb = N * K / 1e9
    print(f"  {name:14s} N={N:6d} K={K:5d}  {best*1e3:7.3f} ms  {gb/best:5.1f} GB/s  "
          f"pack {pack_s*1e3:6.1f} ms")
    return best


def main():
    print("affinity", claim_ai(), flush=True)
    t_qkv = bench("q", NQ, H) + bench("k", KV, H) + bench("v", KV, H)
    t_o = bench("o", H, NQ)
    t_ffn = bench("gate", IM, H) + bench("up", IM, H) + bench("down", H, IM)
    t_lm = bench("lm_head", V, H)
    per_layer = t_qkv + t_o + t_ffn
    total = per_layer * L + t_lm
    print(f"\nper layer {per_layer*1e3:.2f} ms  x{L} = {per_layer*L*1e3:.1f} ms; "
          f"lm_head {t_lm*1e3:.1f} ms")
    print(f"GEMM-only per token: {total*1e3:.1f} ms -> {1/total:.1f} tok/s ceiling")
    print(f"llama.cpp(SpacemiT) 0.6B Q8_0: 30.2 tok/s (33.1 ms/token)")


if __name__ == "__main__":
    main()
