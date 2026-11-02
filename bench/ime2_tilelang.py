"""TileLang -> libime2bridge -> K3 A100 IME2 Q4_0 GEMV."""
import ctypes, time
import numpy as np
import torch
import tilelang
import tilelang.language as T

N, K = 1024, 1024
PB = (N // 32) * (K // 32) * 576
DT = "float32"
TARGET = {"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu",
          "mcpu": "generic-rv64", "mattr": ["+m", "+a", "+f", "+d", "+v"]}
ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so", mode=ctypes.RTLD_GLOBAL)

lib = ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so")
lib.ime2_repack_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

rng = np.random.default_rng(0)
raw = bytearray()
for n in range(N):
    for b in range(K // 32):
        raw += np.float16(rng.uniform(0.001, 0.02)).tobytes() + rng.integers(0, 256, 16, dtype=np.uint8).tobytes()
raw = np.frombuffer(bytes(raw), dtype=np.uint8).copy()
packed = np.zeros(PB, dtype=np.uint8)
assert lib.ime2_repack_q4(raw.ctypes.data, packed.ctypes.data, N, K) == 0

@tilelang.jit(target=TARGET, target_host=TARGET)
def gemv(m=N, k=K, pb=PB):
    @T.prim_func
    def main(P: T.Tensor((pb,), "uint8"), X: T.Tensor((k,), DT), Y: T.Tensor((m,), DT)):
        with T.Kernel(1, threads=1) as bx:
            T.call_extern("int", "ime2_gemv_q4", T.address_of(P[0]), T.address_of(X[0]), T.address_of(Y[0]), m, k)
    return main

k = gemv()
p = torch.from_numpy(packed)
x = torch.randn(K)
y = torch.zeros(N)
k(p, x, y)
print("y0", y[0].item())

for _ in range(3):
    k(p, x, y)
t0 = time.perf_counter()
it = 20
for _ in range(it):
    k(p, x, y)
dt = (time.perf_counter() - t0) / it
bytes_moved = PB
print(f"tilelang->ime2: {dt*1e3:.3f} ms  {bytes_moved/dt/1e9:.2f} GB/s")

t0 = time.perf_counter()
for _ in range(it):
    lib.ime2_gemv_q4(packed.ctypes.data, x.numpy().ctypes.data, y.numpy().ctypes.data, N, K)
dt2 = (time.perf_counter() - t0) / it
print(f"ctypes direct : {dt2*1e3:.3f} ms  {bytes_moved/dt2/1e9:.2f} GB/s")
