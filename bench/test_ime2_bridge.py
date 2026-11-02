"""Correctness test for libime2bridge (repack + IME2 GEMV) against dequant ref."""
import ctypes
import numpy as np

N, K = 1024, 1024
rng = np.random.default_rng(0)

# Build a Q4_0 tensor: blocks of 32 values, 18 bytes each (fp16 d + 16 packed bytes).
nb = K // 32
raw = bytearray()
dvals = []
qs_all = []
for n in range(N):
    for b in range(nb):
        d = np.float16(rng.uniform(0.001, 0.02))
        q = rng.integers(0, 256, size=16, dtype=np.uint8)
        raw += d.tobytes() + q.tobytes()
        dvals.append(float(d))
        qs_all.append(q)
raw = np.frombuffer(bytes(raw), dtype=np.uint8).copy()

packed = np.zeros(N * K * 9 // 16, dtype=np.uint8)  # (N/32)*(K/32)*576
lib = ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so")
lib.ime2_repack_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.ime2_gemv_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
rc = lib.ime2_repack_q4(raw.ctypes.data, packed.ctypes.data, N, K)
print("repack rc", rc)
assert rc == 0

x = rng.standard_normal(K).astype(np.float32)
y = np.zeros(N, dtype=np.float32)
rc = lib.ime2_gemv_q4(packed.ctypes.data, x.ctypes.data, y.ctypes.data, N, K)
print("gemv rc", rc)

# Dequantize reference
W = np.empty((N, K), dtype=np.float32)
for n in range(N):
    for b in range(nb):
        d = dvals[n * nb + b]
        q = qs_all[n * nb + b]
        base = b * 32
        for j in range(16):
            lo = (int(q[j]) & 0x0F) - 8
            hi = (int(q[j]) >> 4) - 8
            W[n, base + j] = d * lo
            W[n, base + 16 + j] = d * hi
ref = W @ x
err = np.abs(y - ref).max()
scale = np.abs(ref).max()
print(f"max_abs_err={err:.6e} rel={err/scale:.3e}")
print("PASS" if err / scale < 2e-2 else "FAIL")
