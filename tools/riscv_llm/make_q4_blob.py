"""Build a Q4_0 blob for the IME2 decode path (K3 A100) from a Q4_0 GGUF.

Why a separate blob: the IME2 GEMV consumes the vendor's 32x32 tile repack
(Q4_0 -> 576-byte tiles), not our planar int8 grid, and it reads half the bytes
per weight.  Measured 19-22 GB/s on A100 against 15.3 GB/s for the int8 RVV
GEMV, so a 0.6B decode step drops from ~43 ms to ~23 ms.

Blob layout (all offsets absolute, 256-byte aligned):
    embed (fp16, V*H*2)          final_norm (fp32, H*4)
    layer_base: L * layer_stride
        rms_attn(f32 H) rms_ffn(f32 H) [q_norm(kv f32) k_norm(kv f32)]
        wq wk wv wo wgate wup wdown   -- each (N/32)*(K/32)*576 bytes
    lm_head packed (V/32)*(H/32)*576  (repacked from token_embd when tied)

    python make_q4_blob.py model-q4_0.gguf out.bin
"""

from __future__ import annotations

import ctypes
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from from_gguf import GGUF, align  # noqa: E402

GGML_TYPE_Q4_0 = 2
BRIDGE = os.environ.get(
    "IME2_BRIDGE", "/work/kubuds/hermes/tl_mahiru/bench/libime2bridge.so")


def q4_nbytes(n: int, k: int) -> int:
    return (n // 32) * (k // 32) * 576


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    src, out = sys.argv[1], sys.argv[2]
    g = GGUF(src)
    names = g.tensors

    def dims_of(nm):
        return names[nm][0]

    H = int(dims_of("token_embd.weight")[0])
    V = int(dims_of("token_embd.weight")[1])
    L = 1 + max(int(n.split(".")[1]) for n in names if n.startswith("blk."))
    blk0 = "blk.0."
    NQ = int(dims_of(blk0 + "attn_q.weight")[1])
    KV = int(dims_of(blk0 + "attn_k.weight")[1])
    IM = int(dims_of(blk0 + "ffn_gate.weight")[1])
    NH = NQ // int(dims_of(blk0 + "attn_q_norm.weight")[0]) if (
        blk0 + "attn_q_norm.weight" in names) else NQ // 128
    HD = 128
    NKV = KV // HD
    has_qk_norm = (blk0 + "attn_q_norm.weight") in names
    tied = "output.weight" not in names
    eps = float(g.kv.get("qwen3.attention.layer_norm_rms_epsilon", 1e-6))
    theta = float(g.kv.get("qwen3.rope.freq_base", 1e6))
    print(f"H={H} L={L} NH={NH} NKV={NKV} HD={HD} IM={IM} V={V} "
          f"qk_norm={has_qk_norm} tied={tied}", flush=True)

    lib = ctypes.CDLL(BRIDGE)
    lib.ime2_repack_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_int, ctypes.c_int]
    lib.ime2_repack_q4.restype = ctypes.c_int

    def raw(nm):
        d, ttype, toff = names[nm]
        n = int(np.prod(d))
        assert ttype == GGML_TYPE_Q4_0, f"{nm}: type {ttype}"
        nb = n // 32 * 18
        start = g.data_off + toff
        return g.mm[start:start + nb], n

    def packed(nm):
        """Q4_0 GGUF tensor -> IME2 tile repack, as bytes."""
        buf, n = raw(nm)
        d = names[nm][0]
        rows, cols = int(d[1]), int(d[0])
        src_arr = np.frombuffer(buf, dtype=np.uint8)
        out_arr = np.zeros(q4_nbytes(rows, cols), dtype=np.uint8)
        rc = lib.ime2_repack_q4(src_arr.ctypes.data, out_arr.ctypes.data,
                                rows, cols)
        assert rc == 0, f"{nm}: repack rc={rc}"
        return out_arr.tobytes()

    def vec(nm, dt=np.float32):
        d, ttype, toff = names[nm]
        n = int(np.prod(d))
        start = g.data_off + toff
        raw_dt = np.float32 if ttype == 0 else np.float16
        return np.frombuffer(g.mm[start:start + n * np.dtype(raw_dt).itemsize],
                             dtype=raw_dt).astype(dt).tobytes()

    # layer stride
    pos = 0
    def _w(nb):
        nonlocal pos
        pos = align(pos, 256) + nb
    _w(H * 4); _w(H * 4)
    if has_qk_norm:
        _w(KV * 4); _w(KV * 4)
    for rows, cols in ((NQ, H), (KV, H), (KV, H), (H, NQ), (IM, H), (IM, H),
                       (H, IM)):
        _w(q4_nbytes(rows, cols))
    layer_stride = align(pos, 256)

    f = open(out, "wb")
    off = {}

    def put(name, data):
        nonlocal f
        pad = (-f.tell()) % 256
        if pad:
            f.write(b"\0" * pad)
        off[name] = f.tell()
        f.write(data)

    # embedding: dequantize Q4_0 rows in chunks to fp16
    ebuf, _ = raw("token_embd.weight")
    qs = np.frombuffer(ebuf, dtype=np.uint8).reshape(V, H // 32, 18)
    embed = np.empty((V, H), dtype=np.float16)
    for r0 in range(0, V, 4096):
        r1 = min(V, r0 + 4096)
        blk = qs[r0:r1]
        d = blk[:, :, 0:2].copy().reshape(-1).view(np.float16)\
            .reshape(r1 - r0, H // 32, 1).astype(np.float32)
        q = blk[:, :, 2:].astype(np.uint8)
        lo = (q & 0x0F).astype(np.float32) - 8.0
        hi = (q >> 4).astype(np.float32) - 8.0
        v = np.concatenate([lo, hi], axis=2) * d
        embed[r0:r1] = v.reshape(r1 - r0, H).astype(np.float16)
    put("embed", embed.tobytes())
    put("final_norm", vec("output_norm.weight"))

    layer_base = None
    layer_off = {}
    for l in range(L):
        p = f"blk.{l}."
        if layer_base is None:
            layer_base = align(f.tell(), 256)
            f.write(b"\0" * (layer_base - f.tell()))
        t0 = f.tell()
        lo = {}
        def w(nm, data):
            pad = (-f.tell()) % 256
            if pad:
                f.write(b"\0" * pad)
            lo[nm] = f.tell() - t0
            f.write(data)
        w("rms_attn", vec(p + "attn_norm.weight"))
        w("rms_ffn", vec(p + "ffn_norm.weight"))
        if has_qk_norm:
            w("q_norm", vec(p + "attn_q_norm.weight"))
            w("k_norm", vec(p + "attn_k_norm.weight"))
        for key, src_name in (("wq", "attn_q.weight"), ("wk", "attn_k.weight"),
                              ("wv", "attn_v.weight"), ("wo", "attn_output.weight"),
                              ("wgate", "ffn_gate.weight"), ("wup", "ffn_up.weight"),
                              ("wdown", "ffn_down.weight")):
            data = packed(p + src_name)
            w(key + ".q", data)
            lo[key + ".s"] = lo[key + ".q"]   # unused by the IME2 kernel
        if l == 0:
            layer_off = lo
        assert f.tell() - t0 <= layer_stride, (f.tell() - t0, layer_stride)
        f.write(b"\0" * (t0 + layer_stride - f.tell()))
        if (l + 1) % 7 == 0 or l == L - 1:
            print(f"  layer {l+1}/{L}", flush=True)

    lm = packed("output.weight" if not tied else "token_embd.weight")
    put("lm", lm)

    total = f.tell()
    f.close()
    meta = dict(
        H=H, L=L, NH=NH, NKV=NKV, HD=HD, IM=IM, V=V, eps=eps, theta=theta,
        has_bias=False, has_qk_norm=has_qk_norm, nbytes=total,
        layer_stride=layer_stride, layer_off=layer_off, layer_base=layer_base,
        embed=off["embed"], embed_nbytes=V * H * 2,
        lm_q=off["lm"], lm_s=off["lm"],
        final_norm=off["final_norm"], nkv_attn=NKV, wbits=4, layout="ime2q4",
    )
    json.dump(meta, open(out + ".json", "w"), indent=1)
    print(f"wrote {out}: {total/1e9:.3f} GB, layer_stride {layer_stride/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
