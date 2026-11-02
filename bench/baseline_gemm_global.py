import sys, time, torch, tilelang
import tilelang.language as T

M = N = K = 64
BM = BN = 32
DT = "float32"
TARGET = {"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu",
          "mcpu": "generic-rv64", "mattr": ["+m", "+a", "+f", "+d"]}


@tilelang.jit(target=TARGET, target_host=TARGET, out_idx=[2])
def matmul(bm=BM, bn=BN, threads=64):
    @T.prim_func
    def main(A: T.Tensor((M, K), DT), B: T.Tensor((K, N), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            for i, j in T.Parallel(bm, bn):
                C[by * bm + i, bx * bn + j] = 0.0
            for k in T.serial(K):
                for i, j in T.Parallel(bm, bn):
                    C[by * bm + i, bx * bn + j] = (
                        C[by * bm + i, bx * bn + j]
                        + A[by * bm + i, k] * B[k, bx * bn + j]
                    )
    return main


torch.manual_seed(0)
a = torch.randn(M, K); b = torch.randn(K, N)
k = matmul()
c = k(a, b)
ref = a @ b
err = (c - ref).abs().max().item(); scale = ref.abs().max().item()
print(f"max_abs_err={err:.6e} rel={err/scale:.3e}")
print("PASS" if err / scale < 1e-4 else "FAIL")
t0 = time.perf_counter(); it = 5
for _ in range(it): k(a, b)
dt = (time.perf_counter() - t0) / it
print(f"latency={dt*1e3:.3f} ms  {2*M*N*K/dt/1e9:.3f} GFLOPS")
