"""Minimal GGUF reader for Qwen3 allq4_0 (Q4_0 + F32 tensors)."""
import struct
import numpy as np


def _rd(f, n):
    b = f.read(n)
    if len(b) != n: raise EOFError
    return b


def _u32(f): return struct.unpack("<I", _rd(f, 4))[0]
def _u64(f): return struct.unpack("<Q", _rd(f, 8))[0]
def _i32(f): return struct.unpack("<i", _rd(f, 4))[0]
def _s(f): return _rd(f, _u64(f)).decode("utf-8", "replace")


def _skip(f, t):
    if t in (0, 1): _rd(f, 1)
    elif t in (2, 3): _rd(f, 2)
    elif t in (4, 5, 6): _rd(f, 4)
    elif t == 7: _rd(f, 1)
    elif t == 8: _s(f)
    elif t in (10, 11, 12): _rd(f, 8)
    elif t == 9:
        et = _u32(f); n = _u64(f)
        for _ in range(n): _skip(f, et)
    else: raise ValueError(t)


def _meta(f, t):
    if t == 8: return _s(f)
    if t == 9:
        et = _u32(f); n = _u64(f)
        return [_meta(f, et) for _ in range(n)]
    if t == 4: return _u32(f)
    if t == 5: return _i32(f)
    if t == 6: return struct.unpack("<f", _rd(f, 4))[0]
    if t == 7: return bool(_rd(f, 1)[0])
    if t == 10: return _u64(f)
    if t == 11: return struct.unpack("<q", _rd(f, 8))[0]
    if t == 12: return struct.unpack("<d", _rd(f, 8))[0]
    raise ValueError(t)


class GGUF:
    def __init__(self, path):
        self.f = open(path, "rb")
        assert _rd(self.f, 4) == b"GGUF"
        self.ver = _u32(self.f)
        nt, nkv = _u64(self.f), _u64(self.f)
        self.meta = {}
        for _ in range(nkv):
            k = _s(self.f); t = _u32(self.f); self.meta[k] = _meta(self.f, t)
        self.tensors = {}
        for _ in range(nt):
            name = _s(self.f); nd = _u32(self.f)
            dims = [_u64(self.f) for _ in range(nd)]
            tt = _u32(self.f); off = _u64(self.f)
            self.tensors[name] = (dims, tt, off)
        self.base = (self.f.tell() + 31) // 32 * 32
        self.tokens = self.meta["tokenizer.ggml.tokens"]
        self.theta = float(self.meta.get("qwen3.rope.freq_base", 10000.0))

    def read(self, name):
        dims, tt, off = self.tensors[name]
        n = 1
        for d in dims: n *= d
        nb = n // 32 * 18 if tt == 2 else n * 4
        self.f.seek(self.base + off)
        return dims, tt, _rd(self.f, nb)

    def deq(self, name):
        dims, tt, data = self.read(name)
        if tt == 0:
            a = np.frombuffer(data, dtype=np.float32).copy()
            return a.reshape(dims[1], dims[0]) if len(dims) == 2 else a
        K, N = dims[0], dims[1]
        q = np.frombuffer(data, dtype=np.uint8).reshape(N, K // 32, 18)
        d = q[:, :, :2].copy().view(np.float16).astype(np.float32).reshape(N, K // 32, 1)
        qs = q[:, :, 2:]
        lo = ((qs & 0xF).astype(np.int16) - 8).astype(np.float32)
        hi = ((qs >> 4).astype(np.int16) - 8).astype(np.float32)
        w = np.empty((N, K // 32, 32), dtype=np.float32)
        w[:, :, :16] = lo * d
        w[:, :, 16:] = hi * d
        return w.reshape(N, K)

    def embed_row(self, tok, K):
        _, _, data = self.read("token_embd.weight")
        r = data[tok * (K // 32) * 18:(tok + 1) * (K // 32) * 18]
        q = np.frombuffer(r, dtype=np.uint8).reshape(K // 32, 18)
        d = q[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        qs = q[:, 2:]
        lo = ((qs & 0xF).astype(np.int16) - 8).astype(np.float32)
        hi = ((qs >> 4).astype(np.int16) - 8).astype(np.float32)
        w = np.empty((K // 32, 32), dtype=np.float32)
        w[:, :16] = lo * d
        w[:, 16:] = hi * d
        return w.reshape(K)
