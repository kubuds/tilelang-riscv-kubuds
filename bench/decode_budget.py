import ctypes, time
import numpy as np

lib = ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so")
lib.ime2_repack_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.ime2_gemv_q4_w.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
rng = np.random.default_rng(0)


def make(N, K):
    nb = K // 32
    d = rng.uniform(0.001, 0.02, size=(N, nb)).astype(np.float16)
    qs = rng.integers(0, 256, size=(N, nb, 16), dtype=np.uint8)
    raw = np.empty((N, nb, 18), dtype=np.uint8)
    raw[:, :, :2] = d.view(np.uint8).reshape(N, nb, 2)
    raw[:, :, 2:] = qs
    p = np.zeros((N // 32) * nb * 576, dtype=np.uint8)
    assert lib.ime2_repack_q4(ctypes.c_char_p(raw.tobytes()), p.ctypes.data, N, K) == 0
    return p


projs = [("q", 2048, 1024), ("k", 1024, 1024), ("v", 1024, 1024), ("o", 1024, 2048),
         ("gate", 3072, 1024), ("up", 3072, 1024), ("down", 1024, 3072)]
layer = 0.0
for name, N, K in projs:
    p = make(N, K)
    x = np.random.randn(K).astype(np.float32)
    y = np.zeros(N, dtype=np.float32)
    lib.ime2_gemv_q4_w(p.ctypes.data, x.ctypes.data, y.ctypes.data, N, K, 4)
    for _ in range(3):
        lib.ime2_gemv_q4_w(p.ctypes.data, x.ctypes.data, y.ctypes.data, N, K, 4)
    t0 = time.perf_counter()
    it = 30
    for _ in range(it):
        lib.ime2_gemv_q4_w(p.ctypes.data, x.ctypes.data, y.ctypes.data, N, K, 4)
    dt = (time.perf_counter() - t0) / it
    layer += dt
    print(f"{name}: {dt*1e3:.3f} ms  {len(p)/dt/1e9:.1f} GB/s")

p = make(151936, 1024)
x = np.random.randn(1024).astype(np.float32)
y = np.zeros(151936, dtype=np.float32)
lib.ime2_gemv_q4_w(p.ctypes.data, x.ctypes.data, y.ctypes.data, 151936, 1024, 4)
for _ in range(3):
    lib.ime2_gemv_q4_w(p.ctypes.data, x.ctypes.data, y.ctypes.data, 151936, 1024, 4)
t0 = time.perf_counter()
for _ in range(10):
    lib.ime2_gemv_q4_w(p.ctypes.data, x.ctypes.data, y.ctypes.data, 151936, 1024, 4)
lm = (time.perf_counter() - t0) / 10
print(f"lm_head: {lm*1e3:.3f} ms  {len(p)/lm/1e9:.1f} GB/s")

total = layer * 28 + lm
print(f"layer(7 gemv)={layer*1e3:.3f} ms  28 layers={layer*28*1e3:.2f} ms  +lm_head={total*1e3:.2f} ms")
print(f"projected decode: {1000/total:.1f} tok/s (attention/norm/sampling excluded)")
