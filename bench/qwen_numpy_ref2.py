import sys
sys.path.insert(0, "/work/mahiru/src/tl/tilelang-0.1.15/bench")
import numpy as np
from gguf_loader import GGUF

MODEL = "/work/mahiru/models/Qwen3-0.6B-allq4_0.gguf"
HID, NH, NKV, HD, FF, NL = 1024, 16, 8, 128, 3072, 28
g = GGUF(MODEL)
theta = g.theta
freq = 1.0 / (theta ** (np.arange(0, HD, 2, dtype=np.float32) / HD))


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


E = g.deq("token_embd.weight")
W = []
for l in range(NL):
    W.append({n: g.deq(f"blk.{l}.{n}.weight") for n in
              ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down",
               "attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm")})
nf = g.deq("output_norm.weight").reshape(-1)

ids = [785, 6722]
cache = [{"k": [], "v": []} for _ in range(NL)]
last = None
for pos, tok in enumerate(ids):
    x = E[tok].copy()
    for l in range(NL):
        d = W[l]
        an = d["attn_norm"].reshape(-1); fn = d["ffn_norm"].reshape(-1)
        qn = d["attn_q_norm"].reshape(-1); kn = d["attn_k_norm"].reshape(-1)
        xn = rms(x, an)
        q = d["attn_q"] @ xn; k = d["attn_k"] @ xn; v = d["attn_v"] @ xn
        q = np.concatenate([rope(rms(q[h*HD:(h+1)*HD], qn), pos) for h in range(NH)])
        k = np.concatenate([rope(rms(k[h*HD:(h+1)*HD], kn), pos) for h in range(NKV)])
        cache[l]["k"].append(k); cache[l]["v"].append(v)
        att = np.empty(NH * HD, dtype=np.float32)
        for h in range(NH):
            kvh = h * NKV // NH
            ks = np.stack([c[kvh*HD:(kvh+1)*HD] for c in cache[l]["k"]])
            vs = np.stack([c[kvh*HD:(kvh+1)*HD] for c in cache[l]["v"]])
            sc = ks @ q[h*HD:(h+1)*HD] / np.sqrt(HD)
            p = np.exp(sc - sc.max()); p /= p.sum()
            att[h*HD:(h+1)*HD] = p @ vs
        x = x + d["attn_output"] @ att
        xn = rms(x, fn)
        gv = d["ffn_gate"] @ xn; uv = d["ffn_up"] @ xn
        x = x + d["ffn_down"] @ ((gv / (1 + np.exp(-gv))) * uv)
    last = x
    print("token", pos, "xmax", np.abs(x).max())

logits = E @ rms(last, nf)
idx = np.argsort(logits)[::-1][:5]
for i in idx:
    print(f"  {int(i):6d} {g.tokens[int(i)]!r} {logits[int(i)]:.3f}")
