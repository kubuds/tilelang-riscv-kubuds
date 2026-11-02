import sys
sys.path.insert(0, "/work/mahiru/src/tl/tilelang-0.1.15/bench")
import ctypes
import numpy as np
from gguf_loader import GGUF

exec(open("/work/mahiru/src/tl/tilelang-0.1.15/bench/qwen_gen.py").read().split("prompt = [785")[0])

lib.qwen_layer_fwd.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p] * 2

g = GGUF("/work/mahiru/models/Qwen3-0.6B-allq4_0.gguf")
E = g.deq("token_embd.weight")
nf2 = g.deq("output_norm.weight").reshape(-1)

KC[:] = 0
VC[:] = 0
ids = [785, 6722]
for pos, t in enumerate(ids):
    xv[:] = embed(t)
    for l in range(NL):
        r = lib.qwen_layer_fwd(xv.ctypes.data, *[m.ctypes.data for m in W[l]],
                               an[l].ctypes.data, fn[l].ctypes.data, qn[l].ctypes.data, kn[l].ctypes.data,
                               cosT.ctypes.data, sinT.ctypes.data, pos, MAXT, KC.ctypes.data, VC.ctypes.data)
        assert r == 0
xn = xv / np.sqrt((xv * xv).mean() + 1e-6) * nf2
logits2 = E @ xn
idx = np.argsort(logits2)[::-1][:5]
print("layer-path top5:")
for i in idx:
    print(f"  {int(i):6d} {g.tokens[int(i)]!r} {logits2[int(i)]:.3f}")
