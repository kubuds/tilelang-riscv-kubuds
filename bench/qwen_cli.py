"""Qwen3-0.6B CLI: text prompt -> GGUF BPE -> single-launch graph -> text."""
import sys
sys.path.insert(0, "/work/mahiru/src/tl/tilelang-0.1.15/bench")
import ctypes
import time
import numpy as np
from gguf_loader import GGUF
from qwen_bpe import QwenBPE

MODEL = "/work/mahiru/models/Qwen3-0.6B-allq4_0.gguf"
HID, NH, NKV, HD, FF, NL, VOCAB, MAXT = 1024, 16, 8, 128, 3072, 28, 151936, 512

lib = ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so")
lib.ime2_repack_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.qwen_graph_fwd.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]

g = GGUF(MODEL)
bpe = QwenBPE(g.tokens, g.meta["tokenizer.ggml.merges"])


def repack(name):
    dims, tt, data = g.read(name)
    K, N = dims[0], dims[1]
    p = np.zeros((N // 32) * (K // 32) * 576, dtype=np.uint8)
    assert lib.ime2_repack_q4(ctypes.c_char_p(data), p.ctypes.data, N, K) == 0
    return p


W, an, fn, qn, kn = [], [], [], [], []
for l in range(NL):
    W.append([repack(f"blk.{l}.{n}.weight") for n in
              ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")])
    an.append(g.deq(f"blk.{l}.attn_norm.weight").reshape(-1))
    fn.append(g.deq(f"blk.{l}.ffn_norm.weight").reshape(-1))
    qn.append(g.deq(f"blk.{l}.attn_q_norm.weight").reshape(-1))
    kn.append(g.deq(f"blk.{l}.attn_k_norm.weight").reshape(-1))
nf = g.deq("output_norm.weight").reshape(-1)
lm = repack("token_embd.weight")
_, _, EMB = g.read("token_embd.weight")


def embed(tok):
    r = EMB[tok * 576:(tok + 1) * 576]
    q = np.frombuffer(r, dtype=np.uint8).reshape(32, 18)
    d = q[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    qs = q[:, 2:]
    lo = ((qs & 0xF).astype(np.int16) - 8).astype(np.float32)
    hi = ((qs >> 4).astype(np.int16) - 8).astype(np.float32)
    w = np.empty((32, 32), dtype=np.float32)
    w[:, :16] = lo * d
    w[:, 16:] = hi * d
    return w.reshape(HID)


freq = 1.0 / (g.theta ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
ang = np.arange(MAXT, dtype=np.float32)[:, None] * freq[None, :]
cosT, sinT = np.cos(ang).astype(np.float32).copy(), np.sin(ang).astype(np.float32).copy()
KC = np.zeros((NL, NKV, MAXT, HD), dtype=np.float32)
VC = np.zeros((NL, NKV, MAXT, HD), dtype=np.float32)
xv = np.zeros(HID, dtype=np.float32)
logits = np.zeros(VOCAB + 1, dtype=np.float32)

ptrs = []
for l in range(NL):
    for m in W[l]: ptrs.append(m.ctypes.data)
    ptrs += [an[l].ctypes.data, fn[l].ctypes.data, qn[l].ctypes.data, kn[l].ctypes.data]
ptrs += [nf.ctypes.data, cosT.ctypes.data, sinT.ctypes.data, lm.ctypes.data,
         xv.ctypes.data, KC.ctypes.data, VC.ctypes.data, logits.ctypes.data]
arr = (ctypes.c_void_p * len(ptrs))(*ptrs)


def sample(lg, temp, topk, topp):
    if temp <= 1e-6:
        return int(np.argmax(lg))
    l = lg / np.float32(temp)
    k = min(max(topk, 1), l.size)
    order = np.argpartition(l, -k)[-k:]
    order = order[np.argsort(l[order])[::-1]]
    p = np.exp(l[order] - l[order].max())
    p /= p.sum()
    c = np.cumsum(p)
    keep = int(np.searchsorted(c, topp) + 1) if topp < 1.0 else len(p)
    keep = max(keep, 1)
    order, p = order[:keep], p[:keep]
    p /= p.sum()
    return int(np.random.choice(order, p=p))


def run(prompt, ngen=24, temp=0.0, topk=1, topp=1.0):
    ids = bpe.encode(prompt)
    KC[:] = 0; VC[:] = 0
    for pos, t in enumerate(ids):
        xv[:] = embed(t)
        assert lib.qwen_graph_fwd(arr, pos, MAXT) == 0
    out = []
    t0 = time.perf_counter()
    for i in range(ngen):
        cur = sample(logits[:VOCAB], temp, topk, topp)
        out.append(cur)
        xv[:] = embed(cur)
        if lib.qwen_graph_fwd(arr, len(ids) + i, MAXT) != 0:
            break
    dt = time.perf_counter() - t0
    return bpe.decode(ids), bpe.decode(out), dt / max(1, ngen)


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    topk = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    topp = float(sys.argv[5]) if len(sys.argv) > 5 else 0.9
    p, o, ms = run(prompt, n, temp, topk, topp)
    print("prompt:", p)
    print("output:", o)
    print(f"{ms*1e3:.2f} ms/token  {1/ms:.1f} tok/s")
