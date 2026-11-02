import sys, time, torch, tilelang
import tilelang.language as T

N = 4096
TARGET = {"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu",
          "mcpu": "generic-rv64", "mattr": ["+m", "+a", "+f", "+d"]}

@tilelang.jit(target=TARGET, target_host=TARGET, out_idx=[2])
def vadd(threads=256):
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                idx = bx * threads + i
                C[idx] = A[idx] + B[idx]
    return main

a = torch.randn(N); b = torch.randn(N)
k = vadd(threads=256)
c = k(a, b)
print("max_err", (c - (a + b)).abs().max().item())
print("PASS" if torch.allclose(c, a + b, atol=1e-5) else "FAIL")
