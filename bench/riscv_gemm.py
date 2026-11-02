"""RISC-V CPU GEMM primitive: T.gemm_cblas (OpenBLAS, row-major f32)."""
import ctypes
import sys
import time

ctypes.CDLL("libopenblas.so.0", mode=ctypes.RTLD_GLOBAL)

import torch
import tilelang
import tilelang.language as T

M = N = K = 512
DT = "float32"
TARGET = {"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu",
          "mcpu": "generic-rv64", "mattr": ["+m", "+a", "+f", "+d", "+v"]}


@tilelang.jit(target=TARGET, target_host=TARGET, out_idx=[2], compile_flags=["-I/usr/include"])
def sgemm(m=M, n=N, k=K):
    @T.prim_func
    def main(A: T.Tensor((m, k), DT), B: T.Tensor((k, n), DT), C: T.Tensor((m, n), DT)):
        with T.Kernel(1, threads=1) as bx:
            T.gemm_cblas(A, B, C, m, n, k)
    return main


torch.manual_seed(0)
a = torch.randn(M, K)
b = torch.randn(K, N)
kernel = sgemm()
c = kernel(a, b)
ref = a @ b
err = (c - ref).abs().max().item()
scale = ref.abs().max().item()
print(f"max_abs_err={err:.6e} rel={err/scale:.3e}")
for _ in range(3):
    kernel(a, b)
t0 = time.perf_counter()
it = 10
for _ in range(it):
    kernel(a, b)
dt = (time.perf_counter() - t0) / it
print(f"latency={dt*1e3:.3f} ms  {2*M*N*K/dt/1e9:.2f} GFLOPS")
print("PASS" if err / scale < 1e-4 else "FAIL")
sys.exit(0 if err / scale < 1e-4 else 1)
