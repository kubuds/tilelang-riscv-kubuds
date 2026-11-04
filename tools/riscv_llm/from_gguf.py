"""Convert a Q8_0 GGUF into the planar blob the RISC-V TileLang engine mmaps.

Why this exists: the HF exporter needs a 16 GB fp16 checkpoint to produce an
8 GB int8 blob.  A Q8_0 GGUF already holds exactly the same 32-element / fp16
scale grid, so the 8-bit data can be copied instead of recomputed (the only
tensor that needs dequantising is the token embedding, which the engine reads
as fp16).

Layout is identical to tools/riscv_llm/export_weights.py; see that file for
the blob description.

    python tools/riscv_llm/from_gguf.py <model.gguf> <out.bin>

GGUF format reference: ggml `gguf` spec v3 (magic "GGUF"), little endian.
Q8_0 block: 2-byte fp16 scale + 32 int8 = 34 bytes.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct

import numpy as np

GGUF_MAGIC = b"GGUF"
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8


def align(n: int, a: int = 256) -> int:
    return (n + a - 1) // a * a


class GGUF:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self._parse_header()

    def _u32(self, off):
        return struct.unpack_from("<I", self.mm, off)[0]

    def _u64(self, off):
        return struct.unpack_from("<Q", self.mm, off)[0]

    def _str(self, off):
        n = self._u64(off)
        return self.mm[off + 8:off + 8 + n].decode("utf-8"), off + 8 + n

    def _parse_header(self):
        assert self.mm[:4] == GGUF_MAGIC, "not a GGUF file"
        self.version = self._u32(4)
        n_tensors = self._u64(8)
        n_kv = self._u64(16)
        off = 24
        self.kv = {}
        for _ in range(n_kv):
            key, off = self._str(off)
            vtype = self._u32(off)
            off += 4
            off = self._read_value(key, vtype, off)
        self.tensors = {}
        for _ in range(n_tensors):
            name, off = self._str(off)
            n_dims = self._u32(off)
            off += 4
            dims = [self._u64(off + 8 * i) for i in range(n_dims)]
            off += 8 * n_dims
            ttype = self._u32(off)
            toff = self._u64(off + 4)
            off += 12
            self.tensors[name] = (dims, ttype, toff)
        al = int(self.kv.get("general.alignment", 32))
        self.data_off = (off + al - 1) // al * al

    def _read_value(self, key, vtype, off):
        # 0 u8,1 i8,2 u16,3 i16,4 u32,5 i32,6 f32,7 bool,8 string,9 array,
        # 10 u64, 11 i64, 12 f64
        if vtype == 8:
            v, off = self._str(off)
            self.kv[key] = v
        elif vtype == 4:
            self.kv[key] = self._u32(off)
            off += 4
        elif vtype == 5:
            self.kv[key] = struct.unpack_from("<i", self.mm, off)[0]
            off += 4
        elif vtype == 6:
            self.kv[key] = struct.unpack_from("<f", self.mm, off)[0]
            off += 4
        elif vtype == 7:
            self.kv[key] = bool(self.mm[off])
            off += 1
        elif vtype == 10:
            self.kv[key] = self._u64(off)
            off += 8
        elif vtype == 9:
            etype = self._u32(off)
            n = self._u64(off + 4)
            off += 12
            for _ in range(n):
                off = self._skip(etype, off)
        else:
            raise NotImplementedError(f"gguf kv type {vtype} ({key})")
        return off

    def _skip(self, vtype, off):
        if vtype == 8:
            _, off = self._str(off)
            return off
        return off + {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                      10: 8, 11: 8, 12: 8}[vtype]

    def tensor_bytes(self, name):
        dims, ttype, toff = self.tensors[name]
        n = int(np.prod(dims))
        if ttype == GGML_TYPE_Q8_0:
            nbytes = n // 32 * 34
        elif ttype == GGML_TYPE_F32:
            nbytes = n * 4
        elif ttype == GGML_TYPE_F16:
            nbytes = n * 2
        else:
            raise NotImplementedError(f"{name}: ggml type {ttype}")
        start = self.data_off + toff
        return self.mm[start:start + nbytes], ttype


class Blob:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.pos = 0
        self.off = {}

    def write(self, name, data, align_to=256):
        pad = align(self.pos, align_to) - self.pos
        if pad:
            self.f.write(b"\0" * pad)
            self.pos += pad
        self.off[name] = self.pos
        b = data.tobytes() if hasattr(data, "tobytes") else data
        self.f.write(b)
        self.pos += len(b)

    def close(self):
        self.f.close()


def split_q8_0(buf, n, k):
    """GGUF Q8_0 stream -> planar q [n,k] int8 and s [n,k/32] fp16."""
    a = np.frombuffer(buf, dtype=np.uint8).reshape(n, k // 32, 34)
    # the block quants are signed; a raw byte view would add 256*scale to every
    # negative value once multiplied downstream
    q = np.ascontiguousarray(a[:, :, 2:]).reshape(n, k).view(np.int8)
    s = np.ascontiguousarray(a[:, :, :2]).reshape(
        n, (k // 32) * 2).view("<f2").reshape(n, k // 32)
    return q, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    args = ap.parse_args()

    g = GGUF(args.gguf)
    names = g.tensors
    kv = g.kv
    H = int(kv["qwen3.embedding_length"])
    L = int(kv["qwen3.block_count"])
    NH = int(kv["qwen3.attention.head_count"])
    NKV = int(kv["qwen3.attention.head_count_kv"])
    HD = int(kv.get("qwen3.attention.key_length", H // NH))
    IM = int(kv["qwen3.feed_forward_length"])
    # GGUF has no vocab_size entry for qwen3; it is token_embd.dim[1].
    V = int(names["token_embd.weight"][0][1])
    eps = float(kv.get("qwen3.attention.layer_norm_rms_epsilon", 1e-6))
    theta = float(kv.get("qwen3.rope.freq_base", 10000.0))
    NQ, KV = NH * HD, NKV * HD
    assert NKV % 4 == 0, f"KV heads {NKV} would need padding; exporter does that"
    has_qk_norm = "blk.0.attn_q_norm.weight" in names
    tied = "output.weight" not in names
    print(f"H={H} L={L} NH={NH} NKV={NKV} HD={HD} IM={IM} V={V} "
          f"qk_norm={has_qk_norm} tied={tied}")

    def mat(nm):
        buf, ttype = g.tensor_bytes(nm)
        dims = names[nm][0]
        rows, cols = int(dims[1]), int(dims[0])
        if ttype == GGML_TYPE_Q8_0:
            return split_q8_0(buf, rows, cols)
        if ttype in (GGML_TYPE_F32, GGML_TYPE_F16):
            dt = np.float32 if ttype == GGML_TYPE_F32 else np.float16
            w = np.frombuffer(buf, dtype=dt).reshape(rows, cols).astype(np.float32)
            return quantize(w)
        raise NotImplementedError(ttype)

    def quantize(w):
        n, k = w.shape
        blocks = w.reshape(n, k // 32, 32)
        amax = np.abs(blocks).max(axis=2)
        amax = np.maximum(amax, 1e-8)
        scale = amax / 127.0
        q = np.clip(np.rint(blocks / scale[:, :, None]), -127, 127).astype(np.int8)
        return q.reshape(n, k), scale.astype(np.float16)

    def vec(nm, dtype=np.float32):
        buf, ttype = g.tensor_bytes(nm)
        dims = names[nm][0]
        dt = {GGML_TYPE_F32: np.float32, GGML_TYPE_F16: np.float16}[ttype]
        return np.frombuffer(buf, dtype=dt).reshape(int(dims[0])).astype(dtype)

    blob = Blob(args.out)

    # embedding as fp16 (the engine copies one row per token)
    ebuf, etype = g.tensor_bytes("token_embd.weight")
    eq, es = split_q8_0(ebuf, V, H) if etype == GGML_TYPE_Q8_0 else (None, None)
    chunk = 4096
    embed = np.empty((V, H), dtype=np.float16)
    for r0 in range(0, V, chunk):
        r1 = min(V, r0 + chunk)
        if eq is not None:
            dq = eq[r0:r1].astype(np.float32) * np.repeat(
                es[r0:r1].astype(np.float32), 32, axis=1)
        else:
            dq = np.frombuffer(ebuf, dtype=np.float16).reshape(V, H)[r0:r1]
        embed[r0:r1] = dq.astype(np.float16)
    blob.write("embed", embed.tobytes())

    # Per-layer stride, computed arithmetically with the same align-to-256 rule
    # the writer applies (the exporter uses a probe Blob; both must agree).
    pos = 0

    def _w(n):
        nonlocal pos
        pos = align(pos, 256) + n

    _w(H * 4)
    _w(H * 4)
    if has_qk_norm:
        _w(HD * 4)
        _w(HD * 4)
    for rows, cols in ((NQ, H), (KV, H), (KV, H), (H, NQ), (IM, H), (IM, H),
                       (H, IM)):
        _w(rows * cols)
        _w(rows * (cols // 32) * 2)
    layer_stride = align(pos, 256)

    blob.write("final_norm", vec("output_norm.weight"))
    layer_base = blob.pos
    for l in range(L):
        p = f"blk.{l}."
        t0 = blob.pos
        blob.write("rms_attn", vec(p + "attn_norm.weight"))
        blob.write("rms_ffn", vec(p + "ffn_norm.weight"))
        if has_qk_norm:
            blob.write("q_norm", vec(p + "attn_q_norm.weight"))
            blob.write("k_norm", vec(p + "attn_k_norm.weight"))
        for nm, src in (("wq", "attn_q.weight"), ("wk", "attn_k.weight"),
                        ("wv", "attn_v.weight"), ("wo", "attn_output.weight"),
                        ("wgate", "ffn_gate.weight"), ("wup", "ffn_up.weight"),
                        ("wdown", "ffn_down.weight")):
            q, s = mat(p + src)
            blob.write(nm + ".q", q.tobytes())
            blob.write(nm + ".s", s.tobytes())
        assert blob.pos - t0 <= layer_stride, (blob.pos - t0, layer_stride)
        if l == 0:
            layer_off = {k: v - t0 for k, v in blob.off.items()
                         if t0 <= v < t0 + layer_stride}
        if l % 6 == 0:
            print(f"  layer {l} done ({blob.pos/1e9:.2f} GB)", flush=True)

    if tied:
        lm_q, lm_s = eq, es
    else:
        lm_q, lm_s = mat("output.weight")
    blob.write("lm_head.q", lm_q.tobytes())
    blob.write("lm_head.s", lm_s.tobytes())
    blob.close()
    total = os.path.getsize(args.out)

    meta = dict(H=H, L=L, NH=NH, NKV=NKV, nkv_attn=NKV, HD=HD, IM=IM, V=V, eps=eps,
                theta=theta, has_bias=False, has_qk_norm=has_qk_norm, tied=tied,
                nbytes=total, layer_stride=layer_stride, wbits=8,
                layer_off=layer_off, layer_base=layer_base, embed=0,
                embed_nbytes=V * H * 2, final_norm=blob.off["final_norm"],
                lm_q=blob.off["lm_head.q"], lm_s=blob.off["lm_head.s"])
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {args.out}: {total/1e9:.3f} GB, layer_stride {layer_stride/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
