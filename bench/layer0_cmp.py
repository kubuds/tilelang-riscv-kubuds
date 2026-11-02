import sys
sys.path.insert(0, "/work/mahiru/src/tl/tilelang-0.1.15/bench")
import ctypes
import numpy as np
from gguf_loader import GGUF

exec(open("/work/mahiru/src/tl/tilelang-0.1.15/bench/qwen_gen.py").read().split("prompt = [785")[0])

g = GGUF("/work/mahiru/models/Qwen3-0.6B-allq4_0.gguf")
theta = g.theta
freq = 1.0 / (theta ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
E = g.deq("token_embd.weight")

d = {n: g.deq(f"blk.0.{n}.weight") for n in
     ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down",
      "attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm")}
nf = g.deq("output_norm.weight").reshape(-1)


def rms(x, w):
    r = 1.0 / np.sqrt((x * x).mean() + 1e-6)
    return x * r * w


def rope(h, pos):
    h = h.copy(); half = HD // 2
    ang = pos * freq
    c, s = np.cos(ang), np.sin(ang)
    a, b = h[:half].copy(), h[half:].copy()
    h[:half] = a * c - b * s
    h[half:] = a * s + b * c
    return h


def np_layer(x, kcache, vcache, pos):
    an = d["attn_norm"].reshape(-1); fn = d["ffn_norm"].reshape(-1)
    qn = d["attn_q_norm"].reshape(-1); kn = d["attn_k_norm"].reshape(-1)
    xn = rms(x, an)
    q = d["attn_q"] @ xn; k = d["attn_k"] @ xn; v = d["attn_v"] @ xn
    q = np.concatenate([rope(rms(q[h*HD:(h+1)*HD], qn), pos) for h in range(NH)])
    k = np.concatenate([rope(rms(k[h*HD:(h+1)*HD], kn), pos) for h in range(NKV)])
    kcache.append(k); vcache.append(v)
    att = np.empty(NH * HD, dtype=np.float32)
    for h in range(NH):
        kvh = h * NKV // NH
        ks = np.stack([c[kvh*HD:(kvh+1)*HD] for c in kcache])
        vs = np.stack([c[kvh*HD:(kvh+1)*HD] for c in vcache])
        sc = ks @ q[h*HD:(h+1)*HD] / np.sqrt(HD)
        p = np.exp(sc - sc.max()); p /= p.sum()
        att[h*HD:(h+1)*HD] = p @ vs
    x = x + d["attn_output"] @ att
    xn = rms(x, fn)
    gv = d["ffn_gate"] @ xn; uv = d["ffn_up"] @ xn
    return x + d["ffn_down"] @ ((gv / (1 + np.exp(-gv))) * uv)


kn_, vn_ = [], []
x0 = E[785].copy(); x0 = np_layer(x0, kn_, vn_, 0)
x1 = E[6722].copy(); x1n = np_layer(x1, kn_, vn_, 1)

lib.qwen_layer_fwd.argtypes=[ctypes.c_void_p]*9+[ctypes.c_void_p]*5+[ctypes.c_int]*2+[ctypes.c_void_p]*2

KCV = np.zeros((NKV, MAXT, HD), dtype=np.float32)
VCV = np.zeros((NKV, MAXT, HD), dtype=np.float32)
xc = embed(785).copy()
lib.qwen_layer_fwd(xc.ctypes.data, *[m.ctypes.data for m in [W[0][0], W[0][1], W[0][2], W[0][3], W[0][4], W[0][5], W[0][6]]],
                   an[0].ctypes.data, fn[0].ctypes.data, qn[0].ctypes.data, kn[0].ctypes.data,
                   cosT.ctypes.data, sinT.ctypes.data, 0, MAXT, KCV.ctypes.data, VCV.ctypes.data)
xc = embed(6722).copy()
lib.qwen_layer_fwd(xc.ctypes.data, *[m.ctypes.data for m in [W[0][0], W[0][1], W[0][2], W[0][3], W[0][4], W[0][5], W[0][6]]],
                   an[0].ctypes.data, fn[0].ctypes.data, qn[0].ctypes.data, kn[0].ctypes.data,
                   cosT.ctypes.data, sinT.ctypes.data, 1, MAXT, KCV.ctypes.data, VCV.ctypes.data)

print("layer0 token1: numpy max", np.abs(x1n).max(), "C max", np.abs(xc).max())
print("max diff", np.abs(x1n - xc).max(), "rel", np.abs(x1n - xc).max() / np.abs(x1n).max())
print("numpy[:4]", x1n[:4])
print("C     [:4]", xc[:4])
