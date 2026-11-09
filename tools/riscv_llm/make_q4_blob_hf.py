"""Q4_0 + IME2 blob straight from an HF safetensors checkpoint.

The GGUF route needs a Q4_0 GGUF first, and llama.cpp refuses to requantize
from Q8_0 without --allow-requantize (and the HF->GGUF converter chokes on
checkpoints whose tokenizer.json has no added_tokens list).  Quantizing here
removes both dependencies: the block layout is 12 lines of arithmetic copied
from ggml's quantize_row_q4_0_ref (MIT), and the IME2 repack is the vendor
kernel.

    python make_q4_blob_hf.py <hf_dir> out.bin
"""

from __future__ import annotations

import ctypes
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from from_gguf import align  # noqa: E402

BRIDGE = os.environ.get(
    "IME2_BRIDGE", "/work/kubuds/hermes/tl_mahiru/bench/libime2bridge.so")
QK = 32


def q4_nbytes(n: int, k: int) -> int:
    return (n // 32) * (k // 32) * 576


def quant_q4_0(w: np.ndarray) -> np.ndarray:
    """[n, k] fp32 -> ggml Q4_0 blocks [n, k/32] of 18 bytes (d + 16 nibbles)."""
    n, k = w.shape
    blk = w.reshape(n, k // QK, QK)
    amax = np.abs(blk)
    idx = amax.argmax(axis=2)
    mx = np.take_along_axis(blk, idx[:, :, None], axis=2)[:, :, 0]
    d = mx / -8.0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(d != 0, 1.0 / d, 0.0)
    xi = np.clip((blk * inv[:, :, None] + 8.5).astype(np.int32), 0, 15)
    out = np.zeros((n, k // QK, 18), dtype=np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(n, k // QK, 2)
    out[:, :, 2:18] = xi[:, :, 0:16] | (xi[:, :, 16:32] << 4)
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    hf, out = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(hf, "config.json")))
    H = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    NH = cfg["num_attention_heads"]
    NKV = cfg.get("num_key_value_heads", NH)
    HD = cfg.get("head_dim", H // NH)
    IM = cfg["intermediate_size"]
    V = cfg["vocab_size"]
    eps = cfg.get("rms_norm_eps", 1e-6)
    theta = cfg.get("rope_theta", 1e6)
    tied = "lm_head.weight" not in cfg
    print(f"H={H} L={L} NH={NH} NKV={NKV} HD={HD} IM={IM} V={V} tied={tied}",
          flush=True)

    from safetensors import safe_open
    idx_path = os.path.join(hf, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wmap = json.load(open(idx_path))["weight_map"]
    else:
        shard = [k for k in sorted(os.listdir(hf)) if k.endswith(".safetensors")][0]
        with safe_open(os.path.join(hf, shard), framework="pt") as fh:
            wmap = {k: shard for k in fh.keys()}
    has_qk_norm = "model.layers.0.self_attn.q_norm.weight" in wmap
    has_bias = "model.layers.0.self_attn.q_proj.bias" in wmap
    nkv_attn = max(NKV, 4)   # the RVV/IME2 kernels want whole 32-row tiles

    cache = {}

    def get(name, dtype=torch.float32):
        shard = wmap.get(name)
        if shard is None:
            raise KeyError(name)
        if shard not in cache:
            cache.clear()
            cache[shard] = safe_open(os.path.join(hf, shard), framework="pt")
        t = cache[shard].get_tensor(name)
        return (t.float() if dtype == torch.float32 else t.to(dtype)).numpy()

    lib = ctypes.CDLL(BRIDGE)
    lib.ime2_repack_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_int, ctypes.c_int]
    lib.ime2_repack_q4.restype = ctypes.c_int

    def packed(w: np.ndarray) -> bytes:
        blocks = quant_q4_0(np.ascontiguousarray(w, dtype=np.float32))
        rows, cols = w.shape
        dst = np.zeros(q4_nbytes(rows, cols), dtype=np.uint8)
        rc = lib.ime2_repack_q4(blocks.ctypes.data, dst.ctypes.data, rows, cols)
        assert rc == 0, f"repack rc={rc}"
        return dst.tobytes()

    pos = 0
    def _w(nb):
        nonlocal pos
        pos = align(pos, 256) + nb
    _w(H * 4); _w(H * 4)
    if has_qk_norm:
        _w(2 * HD * 4)
    if has_bias:   # q, k, v biases share one contiguous block
        _w((NH * HD + 2 * nkv_attn * HD) * 4)
    for rows, cols in ((NH * HD, H), (nkv_attn * HD, H), (nkv_attn * HD, H), (H, NH * HD),
                       (IM, H), (IM, H), (H, IM)):
        _w(q4_nbytes(rows, cols))
    layer_stride = align(pos, 256)

    f = open(out, "wb")
    off = {}

    def put(name, data):
        pad = (-f.tell()) % 256
        if pad:
            f.write(b"\0" * pad)
        off[name] = f.tell()
        f.write(data)

    put("embed", get("model.embed_tokens.weight", torch.float16).tobytes())
    put("final_norm", get("model.norm.weight").tobytes())

    layer_base = None
    layer_off = {}
    for l in range(L):
        p = f"model.layers.{l}."
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

        w("rms_attn", get(p + "input_layernorm.weight").tobytes())
        w("rms_ffn", get(p + "post_attention_layernorm.weight").tobytes())
        if has_qk_norm:
            w("q_norm", get(p + "self_attn.q_norm.weight").tobytes())
            w("k_norm", get(p + "self_attn.k_norm.weight").tobytes())
        if has_bias:
            bq = get(p + "self_attn.q_proj.bias").astype(np.float32)
            bk = get(p + "self_attn.k_proj.bias").astype(np.float32)
            bv = get(p + "self_attn.v_proj.bias").astype(np.float32)
            pad = nkv_attn * HD - bk.shape[0]
            if pad > 0:
                bk = np.concatenate([bk, np.zeros(pad, dtype=np.float32)])
                bv = np.concatenate([bv, np.zeros(pad, dtype=np.float32)])
            w("qkv_bias", np.concatenate([bq, bk, bv]).tobytes())
        for key, nm in (("wq", "self_attn.q_proj.weight"),
                        ("wk", "self_attn.k_proj.weight"),
                        ("wv", "self_attn.v_proj.weight"),
                        ("wo", "self_attn.o_proj.weight"),
                        ("wgate", "mlp.gate_proj.weight"),
                        ("wup", "mlp.up_proj.weight"),
                        ("wdown", "mlp.down_proj.weight")):
            wt = get(p + nm)
            if nm in ("self_attn.k_proj.weight", "self_attn.v_proj.weight") \
                    and wt.shape[0] < nkv_attn * HD:
                pad = np.zeros((nkv_attn * HD - wt.shape[0], wt.shape[1]),
                               dtype=np.float32)
                wt = np.concatenate([wt, pad], axis=0)
            data = packed(wt)
            w(key + ".q", data)
            lo[key + ".s"] = lo[key + ".q"]
        if l == 0:
            layer_off = lo
        assert f.tell() - t0 <= layer_stride
        f.write(b"\0" * (t0 + layer_stride - f.tell()))
        if (l + 1) % 4 == 0 or l == L - 1:
            print(f"  layer {l+1}/{L}", flush=True)

    head = "lm_head.weight" if not tied else "model.embed_tokens.weight"
    put("lm", packed(get(head)))
    total = f.tell()
    f.close()

    # Engine convention: NKV is the *stored* (padded) head count, nkv_attn the
    # real GQA group size.  DS-1.5B has 2 KV heads padded to 4 for the kernels.
    meta = dict(H=H, L=L, NH=NH, NKV=nkv_attn, HD=HD, IM=IM, V=V, eps=eps,
                theta=theta,
                has_bias=has_bias, has_qk_norm=has_qk_norm, nbytes=total,
                layer_stride=layer_stride, layer_off=layer_off,
                layer_base=layer_base, embed=off["embed"], embed_nbytes=V * H * 2,
                lm_q=off["lm"], lm_s=off["lm"], final_norm=off["final_norm"],
                nkv_attn=NKV, wbits=4, layout="ime2q4")
    json.dump(meta, open(out + ".json", "w"), indent=1)
    print(f"wrote {out}: {total/1e9:.3f} GB, layer_stride {layer_stride/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
