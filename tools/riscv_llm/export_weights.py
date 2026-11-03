"""Export a Qwen2/Qwen3 checkpoint into the blob the RISC-V TileLang engine mmaps.

Weight format
-------------
Every 2-D weight is int8, row-major ``[N, K]``, with one fp16 scale per 32
consecutive K elements.  ``q`` and ``scale`` live in separate planes so the
GEMV inner loop reads 32 int8 + 1 half instead of chasing interleaved
metadata.  This is the llama.cpp Q8_0 quantisation grid (32-element blocks),
but stored planar; accuracy is therefore the same class as Q8_0.

The blob is a single flat buffer; ``meta.json`` records every tensor's byte
offset so the TIR program can bake them in as constants.  A per-layer stride
(256-byte aligned) keeps the layer loop in TIR a simple ``l * stride``.

Usage
-----
    python tools/riscv_llm/export_weights.py <hf_dir> <out.bin> [--int4]

--int4 is reserved for the 1.5B model where the int8 weight stream exceeds
the DRAM roofline; the engine only accepts --int8 blobs for now.
"""

from __future__ import annotations

import argparse
import json
import os
import struct

import numpy as np
import torch
from safetensors import safe_open


def align(n: int, a: int = 256) -> int:
    return (n + a - 1) // a * a


class Blob:
    def __init__(self, path: str):
        self.f = open(path, "wb")
        self.pos = 0
        self.off: dict[str, int] = {}
        self.nbytes: dict[str, int] = {}

    def write(self, name: str, data: np.ndarray, align_to: int = 256):
        pad = align(self.pos, align_to) - self.pos
        if pad:
            self.f.write(b"\0" * pad)
            self.pos += pad
        self.off[name] = self.pos
        b = data.tobytes()
        self.f.write(b)
        self.pos += len(b)
        self.nbytes[name] = len(b)

    def close(self):
        self.f.close()


def quantize(w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """[N,K] fp -> int8 q [N,K] + fp16 scales [N,K/32]."""
    w = w.detach().float().contiguous()
    n, k = w.shape
    assert k % 32 == 0, k
    blocks = w.reshape(n, k // 32, 32)
    amax = blocks.abs().amax(dim=2).clamp_min(1e-8)
    scale = amax / 127.0
    q = torch.round(blocks / scale[:, :, None]).clamp(-127, 127).to(torch.int8)
    s16 = scale.to(torch.float16).contiguous()
    return q.reshape(n, k).numpy(), s16.numpy()


def add_weight(blob: Blob, name: str, w: torch.Tensor):
    q, s = quantize(w)
    blob.write(name + ".q", q)
    blob.write(name + ".s", s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("out")
    ap.add_argument("--int8", action="store_true", default=True)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.hf_dir, "config.json")))
    H = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    NH = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    HD = cfg.get("head_dim") or H // NH
    IM = cfg["intermediate_size"]
    V = cfg["vocab_size"]
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    theta = float(cfg.get("rope_theta", 10000.0))
    NQ = NH * HD
    KV = NKV * HD
    # LayerNorm/Loop vectoriser in TVM fails for NKV in {2,3} (scalable-vector
    # lanes() throw in storage_rewrite); NKV multiple of 4 compiles.  Pad the
    # K/V projections to 4 heads with zero rows and keep the real head count
    # in nkv_attn -- attention then ignores the padding, which costs two extra
    # (zero) GEMV rows per layer.
    NKV_PAD = NKV if NKV % 4 == 0 else 4
    KV_PAD = NKV_PAD * HD

    tensors = {}
    with safe_open(os.path.join(args.hf_dir, "model.safetensors"), framework="pt") as f:
        names = list(f.keys())
        for n in names:
            # keep bf16 on disk -> fp32 per tensor on demand
            tensors[n] = f.get_tensor(n)

    has_bias = any(k.endswith("q_proj.bias") for k in names)
    has_qk_norm = any(k.endswith("q_norm.weight") for k in names)
    tied = bool(cfg.get("tie_word_embeddings", False)) or "lm_head.weight" not in names

    blob = Blob(args.out)
    blob.write("embed", tensors["model.embed_tokens.weight"].to(torch.float16).numpy())
    if tied:
        lm = tensors["model.embed_tokens.weight"]
    else:
        lm = tensors["lm_head.weight"]

    # Per-layer layout is identical for every layer -> one stride.
    probe = Blob("/dev/null")
    probe.write("rms_attn", np.zeros(H, np.float32))
    probe.write("rms_ffn", np.zeros(H, np.float32))
    if has_qk_norm:
        probe.write("q_norm", np.zeros(HD, np.float32))
        probe.write("k_norm", np.zeros(HD, np.float32))
    if has_bias:
        probe.write("qkv_bias", np.zeros(NQ + 2 * KV_PAD, np.float32))
    for nm, rows, cols in (("wq", NQ, H), ("wk", KV_PAD, H), ("wv", KV_PAD, H),
                           ("wo", H, NQ), ("wgate", IM, H), ("wup", IM, H),
                           ("wdown", H, IM)):
        probe.write(nm + ".q", np.zeros((rows, cols), np.int8))
        probe.write(nm + ".s", np.zeros((rows, cols // 32), np.float16))
    layer_stride = align(probe.pos)
    probe.close()

    blob.write("final_norm", tensors["model.norm.weight"].float().numpy())
    layer_base = blob.pos
    for l in range(L):
        p = f"model.layers.{l}."
        t0 = blob.pos
        blob.write("rms_attn", tensors[p + "input_layernorm.weight"].float().numpy())
        blob.write("rms_ffn", tensors[p + "post_attention_layernorm.weight"].float().numpy())
        if has_qk_norm:
            blob.write("q_norm", tensors[p + "self_attn.q_norm.weight"].float().numpy())
            blob.write("k_norm", tensors[p + "self_attn.k_norm.weight"].float().numpy())
        if has_bias:
            qb = tensors[p + "self_attn.q_proj.bias"].float()
            kb = tensors[p + "self_attn.k_proj.bias"].float()
            vb = tensors[p + "self_attn.v_proj.bias"].float()
            if NKV_PAD != NKV:
                # Keep the bias layout in step with the padded K/V weights,
                # otherwise the engine's fixed NQ/KV offsets read the wrong
                # rows (symptom: fluent-looking garbage).
                kb = torch.cat([kb, torch.zeros(KV_PAD - KV, dtype=kb.dtype)])
                vb = torch.cat([vb, torch.zeros(KV_PAD - KV, dtype=vb.dtype)])
            b = torch.cat([qb, kb, vb])
            blob.write("qkv_bias", b.numpy())
        add_weight(blob, "wq", tensors[p + "self_attn.q_proj.weight"])
        for nm, src in (("wk", "self_attn.k_proj.weight"),
                        ("wv", "self_attn.v_proj.weight")):
            w = tensors[p + src]
            if NKV_PAD != NKV:
                pad = torch.zeros(NKV_PAD * HD, H, dtype=w.dtype)
                pad[: w.shape[0]] = w
                w = pad
            add_weight(blob, nm, w)
        add_weight(blob, "wo", tensors[p + "self_attn.o_proj.weight"])
        add_weight(blob, "wgate", tensors[p + "mlp.gate_proj.weight"])
        add_weight(blob, "wup", tensors[p + "mlp.up_proj.weight"])
        add_weight(blob, "wdown", tensors[p + "mlp.down_proj.weight"])
        assert blob.pos - t0 <= layer_stride, (blob.pos - t0, layer_stride)
        # local offsets are relative to the layer base; the first layer's
        # offsets are the canonical ones.
        if l == 0:
            layer_off = {k: v - t0 for k, v in blob.off.items()
                         if t0 <= v < t0 + layer_stride}
    lm_q, lm_s = quantize(lm)
    blob.write("lm_head.q", lm_q)
    blob.write("lm_head.s", lm_s)
    total = blob.pos
    blob.close()

    meta = dict(H=H, L=L, NH=NH, NKV=NKV_PAD, nkv_attn=NKV, HD=HD, IM=IM, V=V, eps=eps,
                theta=theta, has_bias=has_bias, has_qk_norm=has_qk_norm,
                tied=tied, nbytes=total, layer_stride=layer_stride,
                layer_off=layer_off, layer_base=layer_base,
                embed=0, embed_nbytes=int(tensors["model.embed_tokens.weight"].numel() * 2),
                final_norm=blob.off["final_norm"],
                lm_q=blob.off["lm_head.q"], lm_s=blob.off["lm_head.s"])
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {args.out}: {total/1e9:.3f} GB, layer_stride {layer_stride}"
          f" ({layer_stride/1e6:.2f} MB), qk_norm={has_qk_norm} bias={has_bias}"
          f" tied={tied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
