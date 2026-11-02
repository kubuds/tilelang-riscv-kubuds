"""lm_head-scale IME2 Q4_0 GEMV: 151936 x 1024."""
import ctypes, time
import numpy as np
import torch
import tilelang
import tilelang.language as T

ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so", mode=ctypes.RTLD_GLOBAL)
lib = ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so")
lib.ime2_repack_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

N, K = 151936, 1024
DT = "float32"
TARGET = {"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu",
          "mcpu": "generic-rv64", "mattr": ["+m", "+a", "+f", "+d", "+v"]}
rng = np.random.default_rng(0)
nb = K // 32
d = rng.uniform(0.001, 0.02, size=(N, nb)).astype(np.float16)
qs = rng.integers(0, 256, size=(N, nb, 16), dtype=np.uint8)
raw = np.empty((N, nb, 18), dtype=np.uint8)
raw[:, :, :2] = d.view(np.uint8).reshape(N, nb, 2)
raw[:, :, 2:] = qs
raw = raw.tobytes()
packed = np.zeros((N // 32) * nb * 576, dtype=np.uint8)
t0 = time.perf_counter()
r = lib.ime2_repack_q4(ctypes.c_char_p(raw), packed.ctypes.data, N, K)
print("repack rc", r, f"{time.perf_counter()-t0:.2f}s", len(packed) / 1e6, "MB")
assert r == 0

@tilelang.jit(target=TARGET, out_idx=[2])
def gemv(n, k):
    pb = (n // 32) * (k // 32) * 576
    @T.prim_func
    def main(P: T.Tensor((pb,), "uint8"), X: T.Tensor((k,), DT), Y: T.Tensor((n,), DT)):
        with T.Kernel(1, threads=1) as bx:
            T.call_extern("int", "ime2_gemv_q4", T.address_of(P[0]), T.address_of(X[0]), T.address_of(Y[0]), n, k)
    return main

k = gemv(N, K)
p = torch.from_numpy(packed)
x = torch.randn(K)
y = k(p, x)
for _ in range(2):
    k(p, x)
t0 = time.perf_counter()
it = 10
for _ in range(it):
    k(p, x)
dt = (time.perf_counter() - t0) / it
print(f"lm_head IME2: {dt*1e3:.3f} ms  {len(packed)/dt/1e9:.2f} GB/s  y0={y[0].item():.4f}")
