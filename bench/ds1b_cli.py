"""DeepSeek-R1-Distill-Qwen-1.5B CLI on the single-launch graph."""
import sys
sys.path.insert(0, "/work/mahiru/src/tl/tilelang-0.1.15/bench")
import ctypes
import time
import numpy as np
from gguf_loader import GGUF
from qwen_bpe import QwenBPE

MODEL = "/work/mahiru/models/deepseek-r1-1.5b/ds1b-allq4_0.gguf"
HID, NH, NKV, HD, FF, NL, VOCAB, MAXT = 1536, 12, 2, 128, 8960, 28, 151936, 512

lib = ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so")
lib.ime2_repack_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.ds1b_graph_fwd.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]

g = GGUF(MODEL)
bpe = QwenBPE(g.tokens, g.meta["tokenizer.ggml.merges"])
print("arch", g.meta.get("general.architecture"), "theta", g.theta)


def repack(name):
    dims, tt, data = g.read(name)
    K, N = dims[0], dims[1]
    p = np.zeros((N // 32) * (K // 32) * 576, dtype=np.uint8)
    assert lib.ime2_repack_q4(ctypes.c_char_p(data), p.ctypes.data, N, K) == 0
    return p


W, bq, bk, bv, an, fn = [], [], [], [], [], []
for l in range(NL):
    W.append([repack(f"blk.{l}.{n}.weight") for n in
              ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")])
    bq.append(g.deq(f"blk.{l}.attn_q.bias").reshape(-1))
    bk.append(g.deq(f"blk.{l}.attn_k.bias").reshape(-1))
    bv.append(g.deq(f"blk.{l}.attn_v.bias").reshape(-1))
    an.append(g.deq(f"blk.{l}.attn_norm.weight").reshape(-1))
    fn.append(g.deq(f"blk.{l}.ffn_norm.weight").reshape(-1))
nf = g.deq("output_norm.weight").reshape(-1)
lm = repack("output.weight")
_, _, EMB = g.read("token_embd.weight")


def embed(tok):
    r = EMB[tok * (HID // 32) * 18:(tok + 1) * (HID // 32) * 18]
    q = np.frombuffer(r, dtype=np.uint8).reshape(HID // 32, 18)
    d = q[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    qs = q[:, 2:]
    lo = ((qs & 0xF).astype(np.int16) - 8).astype(np.float32)
    hi = ((qs >> 4).astype(np.int16) - 8).astype(np.float32)
    w = np.empty((HID // 32, 32), dtype=np.float32)
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
    ptrs += [bq[l].ctypes.data, bk[l].ctypes.data, bv[l].ctypes.data, an[l].ctypes.data, fn[l].ctypes.data]
ptrs += [nf.ctypes.data, cosT.ctypes.data, sinT.ctypes.data, lm.ctypes.data,
         xv.ctypes.data, KC.ctypes.data, VC.ctypes.data, logits.ctypes.data]
arr = (ctypes.c_void_p * len(ptrs))(*ptrs)


def run(prompt, ngen=24):
    ids = bpe.encode(prompt)
    if g.meta.get("tokenizer.ggml.add_bos_token", False):
        ids = [int(g.meta.get("tokenizer.ggml.bos_token_id", 151646))] + ids
    KC[:] = 0; VC[:] = 0
    t0 = time.perf_counter()
    for pos, t in enumerate(ids):
        xv[:] = embed(t)
        assert lib.ds1b_graph_fwd(arr, pos, MAXT) == 0
    pre = time.perf_counter() - t0
    out = []
    t0 = time.perf_counter()
    for i in range(ngen):
        cur = int(np.argmax(logits[:VOCAB]))
        out.append(cur)
        xv[:] = embed(cur)
        if lib.ds1b_graph_fwd(arr, len(ids) + i, MAXT) != 0:
            break
    dt = time.perf_counter() - t0
    return bpe.decode(ids), bpe.decode(out), pre, dt / max(1, ngen)


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    p, o, pre, ms = run(prompt, n)
    print("prompt:", p)
    print("output:", o)
    print(f"prefill {pre*1e3:.1f} ms, decode {ms*1e3:.2f} ms/token  {1/ms:.1f} tok/s")
