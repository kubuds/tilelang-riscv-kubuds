import ctypes, time, torch, tilelang
ctypes.CDLL("libopenblas.so.0", mode=ctypes.RTLD_GLOBAL)
import tilelang.language as T

M = N = K = 512
DT = "float32"
TARGET = {"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu",
          "mcpu": "generic-rv64", "mattr": ["+m", "+a", "+f", "+d", "+v"],
          "mabi": "lp64d"}
FLAGS = ["-I/usr/include"]

@tilelang.jit(target=TARGET, target_host=TARGET, out_idx=[2], compile_flags=FLAGS)
def sgemm():
    @T.prim_func
    def main(A: T.Tensor((M, K), DT), B: T.Tensor((K, N), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(1, threads=1) as bx:
            T.import_source("#include <cblas.h>")
            T.call_extern(
                "void", "cblas_sgemm",
                101, 111, 111, M, N, K,
                1.0,
                T.address_of(A[0, 0]), K,
                T.address_of(B[0, 0]), N,
                0.0,
                T.address_of(C[0, 0]), N,
            )
    return main

torch.manual_seed(0)
a = torch.randn(M, K); b = torch.randn(K, N)
k = sgemm()
c = k(a, b)
ref = a @ b
err = (c - ref).abs().max().item(); scale = ref.abs().max().item()
print(f"max_abs_err={err:.6e} rel={err/scale:.3e}")
for _ in range(3): k(a, b)
t0 = time.perf_counter(); it = 10
for _ in range(it): k(a, b)
dt = (time.perf_counter() - t0) / it
print(f"latency={dt*1e3:.3f} ms  {2*M*N*K/dt/1e9:.2f} GFLOPS")
print("PASS" if err/scale < 1e-4 else "FAIL")
