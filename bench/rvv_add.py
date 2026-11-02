"""RVV 1.0 (VLEN=256) elementwise add through TileLang T.vectorized."""
import ctypes, sys, time
import torch
import tilelang
import tilelang.language as T

N = 1 << 22
DT = "float32"
TARGET = {"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu",
          "mcpu": "generic-rv64", "mattr": ["+m", "+a", "+f", "+d", "+v"]}


@tilelang.jit(target=TARGET, target_host=TARGET, out_idx=[2])
def vadd(vec=8, threads=64):
    block = threads * vec

    @T.prim_func
    def main(A: T.Tensor((N,), DT), B: T.Tensor((N,), DT), C: T.Tensor((N,), DT)):
        with T.Kernel(T.ceildiv(N, block), threads=threads) as bx:
            for t in T.Parallel(threads):
                for v in T.vectorized(vec):
                    idx = bx * block + t * vec + v
                    C[idx] = A[idx] + B[idx]
    return main


torch.manual_seed(0)
a = torch.randn(N)
b = torch.randn(N)
for vec in (1, 4, 8):
    k = vadd(vec=vec, threads=64)
    c = k(a, b)
    ok = torch.allclose(c, a + b, atol=1e-5)
    for _ in range(3):
        k(a, b)
    t0 = time.perf_counter()
    it = 20
    for _ in range(it):
        k(a, b)
    dt = (time.perf_counter() - t0) / it
    gbs = 3 * N * 4 / dt / 1e9
    print(f"vec={vec}: {dt*1e3:.3f} ms  {gbs:.2f} GB/s  {'PASS' if ok else 'FAIL'}")
