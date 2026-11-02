"""28-layer Qwen3-0.6B decode loop through the C layer kernel (weights random Q4_0)."""
import ctypes, time
import numpy as np

lib = ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so")
lib.ime2_repack_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.qwen_layer_fwd.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p] * 2
lib.ime2_gemv_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

HID, NH, NKV, HD, FF, NL = 1024, 16, 8, 128, 3072, 28
MAXT = 256
rng = np.random.default_rng(0)


def q4(N, K):
    nb = K // 32
    d = rng.uniform(0.001, 0.02, size=(N, nb)).astype(np.float16)
    qs = rng.integers(0, 256, size=(N, nb, 16), dtype=np.uint8)
    raw = np.empty((N, nb, 18), dtype=np.uint8)
    raw[:, :, :2] = d.view(np.uint8).reshape(N, nb, 2)
    raw[:, :, 2:] = qs
    p = np.zeros((N // 32) * nb * 576, dtype=np.uint8)
    assert lib.ime2_repack_q4(ctypes.c_char_p(raw.tobytes()), p.ctypes.data, N, K) == 0
    return p


shapes = [(NH * HD, HID), (NKV * HD, HID), (NKV * HD, HID), (HID, NH * HD),
          (FF, HID), (FF, HID), (HID, FF)]
t0 = time.perf_counter()
layers = []
for l in range(NL):
    layers.append([q4(N, K) for (N, K) in shapes])
print("weights built", f"{time.perf_counter()-t0:.1f}s")

norm = lambda n: rng.uniform(0.9, 1.1, n).astype(np.float32)
an = [norm(HID) for _ in range(NL)]
fn = [norm(HID) for _ in range(NL)]
qn = norm(HD)
kn = norm(HD)
freq = 1.0 / (10000 ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
t = np.arange(MAXT, dtype=np.float32)[:, None]
ang = t * freq[None, :]
cosT = np.cos(ang).astype(np.float32).copy()
sinT = np.sin(ang).astype(np.float32).copy()

KC = np.zeros((NKV, MAXT, HD), dtype=np.float32)
VC = np.zeros((NKV, MAXT, HD), dtype=np.float32)
x = rng.standard_normal(HID).astype(np.float32)
lm = q4(151936, HID)
yl = np.zeros(151936, dtype=np.float32)

# warm up one layer
lib.qwen_layer_fwd(x.ctypes.data, *[m.ctypes.data for m in layers[0]],
                   an[0].ctypes.data, fn[0].ctypes.data, qn.ctypes.data, kn.ctypes.data,
                   cosT.ctypes.data, sinT.ctypes.data, 0, MAXT, KC.ctypes.data, VC.ctypes.data)

TOK = 32
t0 = time.perf_counter()
for pos in range(TOK):
    for l in range(NL):
        r = lib.qwen_layer_fwd(x.ctypes.data, *[m.ctypes.data for m in layers[l]],
                               an[l].ctypes.data, fn[l].ctypes.data, qn.ctypes.data, kn.ctypes.data,
                               cosT.ctypes.data, sinT.ctypes.data, pos, MAXT, KC.ctypes.data, VC.ctypes.data)
        assert r == 0, r
    lib.ime2_gemv_q4(lm.ctypes.data, x.ctypes.data, yl.ctypes.data, 151936, HID)
dt = time.perf_counter() - t0
print(f"{TOK} tokens in {dt*1e3:.1f} ms -> {TOK/dt:.1f} tok/s ({dt/TOK*1e3:.2f} ms/token)")
