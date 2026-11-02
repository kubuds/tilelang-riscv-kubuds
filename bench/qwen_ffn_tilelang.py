"""Qwen3-style FFN block on K3 X100, assembled from TileLang kernels.

gate/up are two cblas GEMMs, activation is a TileLang elementwise kernel,
down is another cblas GEMM. Shapes: hidden=1024, ffn=3072 (Qwen3-0.6B).
"""
import ctypes, time
import torch
import tilelang
import tilelang.language as T

ctypes.CDLL("libopenblas.so.0", mode=ctypes.RTLD_GLOBAL)

HID, FFN = 1024, 3072
DT = "float32"
TARGET = {"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu",
          "mcpu": "generic-rv64", "mattr": ["+m", "+a", "+f", "+d", "+v"]}


@tilelang.jit(target=TARGET, out_idx=[2], compile_flags=["-I/usr/include"])
def gemm(m, n, k):
    @T.prim_func
    def main(A: T.Tensor((m, k), DT), B: T.Tensor((k, n), DT), C: T.Tensor((m, n), DT)):
        with T.Kernel(1, threads=1) as bx:
            T.gemm_cblas(A, B, C, m, n, k)
    return main


@tilelang.jit(target=TARGET, out_idx=[2])
def silu_mul(n):
    @T.prim_func
    def main(G: T.Tensor((n,), DT), U: T.Tensor((n,), DT), O: T.Tensor((n,), DT)):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
            for i in T.Parallel(256):
                idx = bx * 256 + i
                O[idx] = (G[idx] / (1.0 + T.exp(-G[idx]))) * U[idx]
    return main


def run(m, iters=10):
    torch.manual_seed(0)
    x = torch.randn(m, HID)
    w1 = torch.randn(HID, FFN)
    w3 = torch.randn(HID, FFN)
    w2 = torch.randn(FFN, HID)
    kg = gemm(m, FFN, HID)
    kd = gemm(m, HID, FFN)
    ks = silu_mul(m * FFN)
    ref = (torch.nn.functional.silu(x @ w1) * (x @ w3)) @ w2

    def fwd():
        g = kg(x, w1)
        u = kg(x, w3)
        s = ks(g.reshape(-1), u.reshape(-1)).reshape(m, FFN)
        return kd(s, w2)

    out = fwd()
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    for _ in range(2):
        fwd()
    t0 = time.perf_counter()
    for _ in range(iters):
        fwd()
    dt = (time.perf_counter() - t0) / iters
    flops = 2 * m * HID * FFN * 2 + 2 * m * FFN * HID
    print(f"m={m}: {dt*1e3:.3f} ms  {flops/dt/1e9:.2f} GFLOPS  rel_err={err:.3e}")


for m in (1, 8, 32):
    run(m)
