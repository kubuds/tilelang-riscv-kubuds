"""Whole-model single-launch decode: qwen_graph_fwd (28 layers in one Spert launch)."""
import ctypes, time
import numpy as np

lib = ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so")
lib.ime2_repack_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.qwen_graph_fwd.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]

HID, NH, NKV, HD, FF, NL, VOCAB = 1024, 16, 8, 128, 3072, 28, 151936
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


shapes = [(NH * HD, HID), (NKV * HD, HID), (NKV * HD, HID), (HID, NH * HD), (FF, HID), (FF, HID), (HID, FF)]
t0 = time.perf_counter()
weights = [[q4(N, K) for (N, K) in shapes] for _ in range(NL)]
print("weights built", f"{time.perf_counter()-t0:.1f}s")

norm = lambda n: rng.uniform(0.9, 1.1, n).astype(np.float32)
an = [norm(HID) for _ in range(NL)]
fn = [norm(HID) for _ in range(NL)]
qn, kn = norm(HD), norm(HD)
freq = 1.0 / (10000 ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
ang = np.arange(MAXT, dtype=np.float32)[:, None] * freq[None, :]
cosT, sinT = np.cos(ang).astype(np.float32).copy(), np.sin(ang).astype(np.float32).copy()
x = rng.standard_normal(HID).astype(np.float32)
KC = np.zeros((NKV, MAXT, HD), dtype=np.float32)
VC = np.zeros((NKV, MAXT, HD), dtype=np.float32)
lm = q4(VOCAB, HID)
logits = np.zeros(VOCAB + 1, dtype=np.float32)

ptrs = []
for l in range(NL):
    for m in weights[l]:
        ptrs.append(m.ctypes.data)
    ptrs.append(an[l].ctypes.data)
    ptrs.append(fn[l].ctypes.data)
for extra in (qn, kn, cosT, sinT, lm, x, KC, VC, logits):
    ptrs.append(extra.ctypes.data)
arr = (ctypes.c_void_p * len(ptrs))(*ptrs)

rc = lib.qwen_graph_fwd(arr, 0, MAXT)
print("rc", rc, "argmax", int(logits[VOCAB]))

TOK = 32
t0 = time.perf_counter()
for pos in range(TOK):
    rc = lib.qwen_graph_fwd(arr, pos, MAXT)
    assert rc == 0, rc
dt = time.perf_counter() - t0
print(f"{TOK} tokens in {dt*1e3:.1f} ms -> {TOK/dt:.1f} tok/s ({dt/TOK*1e3:.2f} ms/token)")
print("last argmax", int(logits[VOCAB]))
