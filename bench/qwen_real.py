"""Run one real Qwen3-0.6B token through the single-launch graph (allq4_0 GGUF)."""
import ctypes, struct, sys, time
import numpy as np

MODEL = "/work/mahiru/models/Qwen3-0.6B-allq4_0.gguf"
lib = ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so")
lib.ime2_repack_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.qwen_graph_fwd.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]

HID, NH, NKV, HD, FF, NL, VOCAB, MAXT = 1024, 16, 8, 128, 3072, 28, 151936, 256


def rd(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError
    return b


def u32(f): return struct.unpack("<I", rd(f, 4))[0]
def u64(f): return struct.unpack("<Q", rd(f, 8))[0]
def i32(f): return struct.unpack("<i", rd(f, 4))[0]
def f32v(f): return struct.unpack("<f", rd(f, 4))[0]
def s(f): return rd(f, u64(f)).decode("utf-8", "replace")


def skip(f, t):
    if t in (0, 1): rd(f, 1)
    elif t in (2, 3): rd(f, 2)
    elif t in (4, 5, 6): rd(f, 4)
    elif t == 7: rd(f, 1)
    elif t == 8: s(f)
    elif t == 10: rd(f, 8)
    elif t == 11: rd(f, 8)
    elif t == 12: rd(f, 8)
    elif t in (9,):
        et = u32(f); n = u64(f)
        for _ in range(n): skip(f, et)
    else: raise ValueError(t)


def read_meta(f, t):
    if t == 8: return s(f)
    if t == 9:
        et = u32(f); n = u64(f)
        return [read_meta(f, et) for _ in range(n)]
    if t == 4: return u32(f)
    if t == 5: return i32(f)
    if t == 6: return f32v(f)
    if t == 7: return bool(rd(f, 1)[0])
    if t == 10: return u64(f)
    if t == 11: return struct.unpack("<q", rd(f, 8))[0]
    if t == 12: return struct.unpack("<d", rd(f, 8))[0]
    if t == 0: return rd(f, 1)[0]
    if t == 1: return struct.unpack("<b", rd(f, 1))[0]
    if t == 2: return struct.unpack("<H", rd(f, 2))[0]
    if t == 3: return struct.unpack("<h", rd(f, 2))[0]
    raise ValueError(t)


with open(MODEL, "rb") as f:
    assert rd(f, 4) == b"GGUF"
    ver, nt, nkv = u32(f), u64(f), u64(f)
    meta = {}
    for _ in range(nkv):
        k = s(f); t = u32(f)
        meta[k] = read_meta(f, t)
    tensors = {}
    for _ in range(nt):
        name = s(f); nd = u32(f)
        dims = [u64(f) for _ in range(nd)]
        tt = u32(f); off = u64(f)
        tensors[name] = (dims, tt, off)
    pos = f.tell(); base = (pos + 31) // 32 * 32
    tokens = meta["tokenizer.ggml.tokens"]
    theta = float(meta.get("qwen3.rope.freq_base", meta.get("llama.rope.freq_base", 10000.0)))
    print("tokens", len(tokens), "tensors", len(tensors), "ver", ver)

    def read_tensor(name):
        dims, tt, off = tensors[name]
        n = 1
        for d in dims: n *= d
        nb = n // 32 * 18 if tt == 2 else n * 4 if tt == 0 else n * 2 if tt == 1 else None
        assert nb, (name, tt)
        f.seek(base + off)
        return dims, tt, rd(f, nb)

    packed = {}
    def repack_tensor(name):
        dims, tt, data = read_tensor(name)
        assert tt == 2 and len(dims) == 2, (name, dims, tt)
        K, N = dims[0], dims[1]
        p = np.zeros((N // 32) * (K // 32) * 576, dtype=np.uint8)
        r = lib.ime2_repack_q4(ctypes.c_char_p(data), p.ctypes.data, N, K)
        assert r == 0, (name, r)
        return p

    def norm(name):
        dims, tt, data = read_tensor(name)
        assert tt == 0, (name, tt)
        return np.frombuffer(data, dtype=np.float32).copy()

    t0 = time.perf_counter()
    W = []
    an, fn, qn, kn = [], [], [], []
    for l in range(NL):
        W.append([repack_tensor(f"blk.{l}.{n}.weight") for n in
                  ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")])
        an.append(norm(f"blk.{l}.attn_norm.weight"))
        fn.append(norm(f"blk.{l}.ffn_norm.weight"))
        qn.append(norm(f"blk.{l}.attn_q_norm.weight"))
        kn.append(norm(f"blk.{l}.attn_k_norm.weight"))
    nf = norm("output_norm.weight")
    lm = repack_tensor("token_embd.weight")
    print("weights loaded", f"{time.perf_counter()-t0:.1f}s")

    # embedding row for a probe token
    dims, tt, data = read_tensor("token_embd.weight")
    K, V = dims[0], dims[1]
    for probe in ("The", " Paris", "Hello"):
        ids = [i for i, t in enumerate(tokens) if t == probe]
        if ids:
            tok = ids[0]
            break
    else:
        tok = 0
    row = data[tok * (K // 32) * 18:(tok + 1) * (K // 32) * 18]
    d = np.frombuffer(row[:2], dtype=np.float16)[0].astype(np.float32)
    q = np.frombuffer(row[2:], dtype=np.uint8)
    lo = (q & 0xF).astype(np.int16) - 8
    hi = (q >> 4).astype(np.int16) - 8
    xv = np.empty(K, dtype=np.float32)
    xv[0::32] = 0
    for b in range(K // 32):
        d = np.frombuffer(row[b * 18:b * 18 + 2], dtype=np.float16)[0].astype(np.float32)
        xv[b * 32:b * 32 + 16] = d * lo[b * 16:(b + 1) * 16]
        xv[b * 32 + 16:b * 32 + 32] = d * hi[b * 16:(b + 1) * 16]

freq = 1.0 / (theta ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
ang = np.arange(MAXT, dtype=np.float32)[:, None] * freq[None, :]
cosT, sinT = np.cos(ang).astype(np.float32).copy(), np.sin(ang).astype(np.float32).copy()
KC = np.zeros((NL, NKV, MAXT, HD), dtype=np.float32)
VC = np.zeros((NL, NKV, MAXT, HD), dtype=np.float32)
logits = np.zeros(VOCAB + 1, dtype=np.float32)

ptrs = []
for l in range(NL):
    for m in W[l]: ptrs.append(m.ctypes.data)
    ptrs += [an[l].ctypes.data, fn[l].ctypes.data, qn[l].ctypes.data, kn[l].ctypes.data]
ptrs += [nf.ctypes.data, cosT.ctypes.data, sinT.ctypes.data, lm.ctypes.data,
         xv.ctypes.data, KC.ctypes.data, VC.ctypes.data, logits.ctypes.data]
arr = (ctypes.c_void_p * len(ptrs))(*ptrs)

t0 = time.perf_counter()
rc = lib.qwen_graph_fwd(arr, 0, MAXT)
print("rc", rc, f"{(time.perf_counter()-t0)*1e3:.1f} ms")
idx = np.argsort(logits[:VOCAB])[::-1][:8]
print("probe token", tok, repr(tokens[tok]))
for i in idx:
    print(f"  {int(i):6d} {tokens[int(i)]!r} {logits[int(i)]:.3f}")
