import ctypes, time
import numpy as np

lib = ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so")
lib.qwen_attn_fwd.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5

NH, NKV, HD, MAXT = 16, 8, 128, 2048
rng = np.random.default_rng(0)
q = rng.standard_normal((NH, HD)).astype(np.float32)
KC = rng.standard_normal((NKV, MAXT, HD)).astype(np.float32)
VC = rng.standard_normal((NKV, MAXT, HD)).astype(np.float32)
out = np.zeros((NH, HD), dtype=np.float32)

for seq in (128, 512, 1024):
    pos = seq - 1
    rc = lib.qwen_attn_fwd(q.ctypes.data, KC.ctypes.data, VC.ctypes.data, out.ctypes.data,
                           NH, NKV, HD, pos, MAXT)
    sref = np.stack([q[h] @ KC[h * NKV // NH, :seq].T for h in range(NH)]) / np.sqrt(HD)
    pref = np.exp(sref - sref.max(-1, keepdims=True))
    pref /= pref.sum(-1, keepdims=True)
    oref = np.stack([pref[h] @ VC[h * NKV // NH, :seq] for h in range(NH)])
    err = np.abs(out - oref).max() / np.abs(oref).max()
    for _ in range(3):
        lib.qwen_attn_fwd(q.ctypes.data, KC.ctypes.data, VC.ctypes.data, out.ctypes.data, NH, NKV, HD, pos, MAXT)
    t0 = time.perf_counter()
    it = 50
    for _ in range(it):
        lib.qwen_attn_fwd(q.ctypes.data, KC.ctypes.data, VC.ctypes.data, out.ctypes.data, NH, NKV, HD, pos, MAXT)
    dt = (time.perf_counter() - t0) / it
    print(f"seq={seq}: {dt*1e3:.3f} ms  rel_err={err:.2e}")
