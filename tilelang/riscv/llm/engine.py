"""TileLang inference engine for RISC-V (one TIR program per layer + head).

Why this shape
--------------
A decode step is ~10 operators per layer.  Calling one compiled kernel per
operator costs 280 Python/FFI round trips per token, which on this board is the
same order as the GEMV time itself.  So each *layer* is one prim_func: the
operator sequence is TIR, every operator is a ``T.call_extern`` into the RVV
primitives in ``src/tl_templates/riscv/llm.h``, and the Python side only loops
over layers (29 calls per token, not 280).

Why one layer per program and not the whole model: TVM's
``PointerValueTypeRewrite`` throws ``Can't fetch the lanes of a scalable
vector`` when the same TIR contains the layer loop with L>=2 at the 1.5B
shapes (H=1536, IM=8960).  L=1 lowers fine; see docs/RISCV_BACKEND.md.  The
per-layer split also keeps register and compile pressure flat.

All tensor offsets inside the flat weight blob are baked in from meta.json.

The embedding lookup stays in Python: it is one 2 KB row copy per token, and
keeping it out of the layer program means the layer program is identical for
every layer (only BASE changes).
"""

from __future__ import annotations

import os

import json
import time
from dataclasses import dataclass

import numpy as np
import tilelang
import tilelang.language as T
from tilelang import tvm

RISCV_TARGET = {"kind": "riscv", "march": "rv64gcv", "mabi": "lp64d",
                "vlen": 256}


@dataclass
class Meta:
    H: int
    L: int
    NH: int
    NKV: int
    HD: int
    IM: int
    V: int
    eps: float
    theta: float
    has_bias: bool
    has_qk_norm: bool
    nbytes: int
    layer_stride: int
    layer_off: dict
    layer_base: int
    embed: int
    embed_nbytes: int
    lm_q: int
    lm_s: int
    final_norm: int
    nkv_attn: int = 0
    wbits: int = 8
    layout: str = "q8planar"   # "q8planar" (our RVV) or "ime2q4" (A100 IME2)

    @property
    def NQ(self):
        return self.NH * self.HD

    @property
    def KV(self):
        return self.NKV * self.HD


def _gemv_name(m: Meta, batched: bool = False) -> str:
    """GEMV symbol for this blob layout.

    ime2q4 blobs hold the vendor's 32x32 Q4_0 tiles and are consumed by the A100
    IME2 matrix engine (4 spert workers).  That kernel has no M=8 form, so
    batched prefill stays on the RVV path and is only used for int8 blobs.
    """
    if m.layout == "ime2q4":
        return "tl_riscv_gemv_q4_ime2"
    if batched:
        return "tl_riscv_gemv_q4" if m.wbits == 4 else "tl_riscv_gemv_q8_m8"
    return "tl_riscv_gemv_q4" if m.wbits == 4 else "tl_riscv_gemv_q8"


def _call(name, *args):
    """Return the extern call expression.

    Returning it matters: TileLang's eager builder only emits a statement when
    the traced expression reaches ``__tb.eval``.  A helper that calls
    T.call_extern and returns None evaluates the call in Python and throws the
    expression away, which silently compiles an empty kernel.
    """
    return T.call_extern("int32", name, *args)


def build_step(m: Meta, maxseq: int):
    H, L, NH, NKV, HD, IM, V = m.H, m.L, m.NH, m.NKV, m.HD, m.IM, m.V
    NQ, KVD = m.NQ, m.KV
    eps, LS = m.eps, m.layer_stride
    o = m.layer_off
    scale = 1.0 / (HD ** 0.5)
    GEMV = _gemv_name(m)

    @T.prim_func
    def step(
        W: T.Tensor((m.nbytes,), "uint8"),
        CS: T.Tensor((maxseq, HD // 2), "float32"),
        SN: T.Tensor((maxseq, HD // 2), "float32"),
        KC: T.Tensor((L * NKV * maxseq * HD,), "float32"),
        VC: T.Tensor((L * NKV * maxseq * HD,), "float32"),
        X: T.Tensor((H,), "float32"),
        XB: T.Tensor((H,), "float32"),
        Q: T.Tensor((NQ,), "float32"),
        KB: T.Tensor((KVD,), "float32"),
        VB: T.Tensor((KVD,), "float32"),
        ATT: T.Tensor((NQ,), "float32"),
        GU: T.Tensor((2 * IM,), "float32"),
        IDS: T.Tensor((1,), "int32"),
        POS: T.Tensor((1,), "int32"),
        OUT: T.Tensor((1,), "int32"),
        LOGITS: T.Tensor((V,), "float32"),
    ):
        with T.Kernel(1):
            _call("tl_riscv_embed", H, T.address_of(W[T.cast(m.embed, 'int64')]), IDS[0],
                  T.address_of(X[0]))
            for l in range(L):
                b = m.layer_base + l * LS
                p = POS[0]
                _call("tl_riscv_rmsnorm", H, T.address_of(X[0]),
                      T.address_of(W[T.cast(b + o["rms_attn"], 'int64')]), eps, T.address_of(XB[0]))
                _call(GEMV, NQ, H, T.address_of(W[T.cast(b + o["wq.q"], 'int64')]),
                      T.address_of(W[T.cast(b + o["wq.s"], 'int64')]), T.address_of(XB[0]),
                      T.address_of(Q[0]))
                _call(GEMV, KVD, H, T.address_of(W[T.cast(b + o["wk.q"], 'int64')]),
                      T.address_of(W[T.cast(b + o["wk.s"], 'int64')]), T.address_of(XB[0]),
                      T.address_of(KB[0]))
                _call(GEMV, KVD, H, T.address_of(W[T.cast(b + o["wv.q"], 'int64')]),
                      T.address_of(W[T.cast(b + o["wv.s"], 'int64')]), T.address_of(XB[0]),
                      T.address_of(VB[0]))
                if m.has_bias:
                    _call("tl_riscv_add_u8", NQ, T.address_of(Q[0]),
                          T.address_of(W[T.cast(b + o["qkv_bias"], 'int64')]))
                    _call("tl_riscv_add_u8", KVD, T.address_of(KB[0]),
                          T.address_of(W[T.cast(b + o["qkv_bias"] + NQ * 4, 'int64')]))
                    _call("tl_riscv_add_u8", KVD, T.address_of(VB[0]),
                          T.address_of(W[T.cast(b + o["qkv_bias"] + (NQ + KVD) * 4, 'int64')]))
                for h in range(NH):
                    if m.has_qk_norm:
                        _call("tl_riscv_head_norm", HD,
                              T.address_of(Q[h * HD]),
                              T.address_of(W[T.cast(b + o["q_norm"], 'int64')]), eps)
                    _call("tl_riscv_rope", HD, T.address_of(Q[h * HD]),
                          T.address_of(CS[p, 0]), T.address_of(SN[p, 0]))
                for h in range(NKV):
                    if m.has_qk_norm:
                        _call("tl_riscv_head_norm", HD,
                              T.address_of(KB[h * HD]),
                              T.address_of(W[T.cast(b + o["k_norm"], 'int64')]), eps)
                    _call("tl_riscv_rope", HD, T.address_of(KB[h * HD]),
                          T.address_of(CS[p, 0]), T.address_of(SN[p, 0]))
                for h in range(NKV):
                    _call("tl_riscv_copy", HD,
                          T.address_of(KC[((l * NKV + h) * maxseq + p) * HD]),
                          T.address_of(KB[h * HD]))
                    _call("tl_riscv_copy", HD,
                          T.address_of(VC[((l * NKV + h) * maxseq + p) * HD]),
                          T.address_of(VB[h * HD]))
                _call("tl_riscv_attn_decode", NH, m.nkv_attn, HD, p, maxseq * HD,
                      scale, T.address_of(Q[0]),
                      T.address_of(KC[l * NKV * maxseq * HD]),
                      T.address_of(VC[l * NKV * maxseq * HD]),
                      T.address_of(ATT[0]))
                _call(GEMV, H, NQ, T.address_of(W[T.cast(b + o["wo.q"], 'int64')]),
                      T.address_of(W[T.cast(b + o["wo.s"], 'int64')]), T.address_of(ATT[0]),
                      T.address_of(XB[0]))
                _call("tl_riscv_add", H, T.address_of(X[0]),
                      T.address_of(XB[0]))

                _call("tl_riscv_rmsnorm", H, T.address_of(X[0]),
                      T.address_of(W[T.cast(b + o["rms_ffn"], 'int64')]), eps, T.address_of(XB[0]))
                _call(GEMV, IM, H, T.address_of(W[T.cast(b + o["wgate.q"], 'int64')]),
                      T.address_of(W[T.cast(b + o["wgate.s"], 'int64')]), T.address_of(XB[0]),
                      T.address_of(GU[0]))
                _call(GEMV, IM, H, T.address_of(W[T.cast(b + o["wup.q"], 'int64')]),
                      T.address_of(W[T.cast(b + o["wup.s"], 'int64')]), T.address_of(XB[0]),
                      T.address_of(GU[IM]))
                _call("tl_riscv_silu_mul", IM, T.address_of(GU[0]),
                      T.address_of(GU[IM]), T.address_of(GU[0]))
                _call(GEMV, H, IM, T.address_of(W[T.cast(b + o["wdown.q"], 'int64')]),
                      T.address_of(W[T.cast(b + o["wdown.s"], 'int64')]), T.address_of(GU[0]),
                      T.address_of(XB[0]))
                _call("tl_riscv_add", H, T.address_of(X[0]),
                      T.address_of(XB[0]))

            _call("tl_riscv_rmsnorm", H, T.address_of(X[0]),
                  T.address_of(W[T.cast(m.final_norm, 'int64')]), eps, T.address_of(XB[0]))
            _call(GEMV, V, H, T.address_of(W[T.cast(m.lm_q, 'int64')]),
                  T.address_of(W[T.cast(m.lm_s, 'int64')]), T.address_of(XB[0]),
                  T.address_of(LOGITS[0]))
            _call("tl_riscv_argmax", V, T.address_of(LOGITS[0]),
                  T.address_of(OUT[0]))

    return step


def build_layer(m: Meta, maxseq: int):
    H, NH, NKV, HD, IM = m.H, m.NH, m.NKV, m.HD, m.IM
    NQ, KVD = m.NQ, m.KV
    eps = m.eps
    o = m.layer_off
    scale = 1.0 / (HD ** 0.5)
    GEMV = _gemv_name(m)

    @T.prim_func
    def layer(
        W: T.Tensor((m.nbytes,), "uint8"),
        CS: T.Tensor((maxseq, HD // 2), "float32"),
        SN: T.Tensor((maxseq, HD // 2), "float32"),
        KC: T.Tensor((m.L * NKV * maxseq * HD,), "float32"),
        VC: T.Tensor((m.L * NKV * maxseq * HD,), "float32"),
        X: T.Tensor((H,), "float32"),
        XB: T.Tensor((H,), "float32"),
        Q: T.Tensor((NQ,), "float32"),
        KB: T.Tensor((KVD,), "float32"),
        VB: T.Tensor((KVD,), "float32"),
        ATT: T.Tensor((NQ,), "float32"),
        GU: T.Tensor((2 * IM,), "float32"),
        POS: T.Tensor((1,), "int32"),
        BASE: T.Tensor((1,), "int64"),
        LI: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            # Widened before the add: the 1.5B blob is 2.1 GB and an int32
            # element index wraps.
            b = BASE[0]
            p = POS[0]
            l = LI[0]
            _call("tl_riscv_rmsnorm", H, T.address_of(X[0]),
                  T.address_of(W[T.cast(b + o["rms_attn"], "int64")]), eps,
                  T.address_of(XB[0]))
            _call(GEMV, NQ, H,
                  T.address_of(W[T.cast(b + o["wq.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wq.s"], "int64")]),
                  T.address_of(XB[0]), T.address_of(Q[0]))
            _call(GEMV, KVD, H,
                  T.address_of(W[T.cast(b + o["wk.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wk.s"], "int64")]),
                  T.address_of(XB[0]), T.address_of(KB[0]))
            _call(GEMV, KVD, H,
                  T.address_of(W[T.cast(b + o["wv.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wv.s"], "int64")]),
                  T.address_of(XB[0]), T.address_of(VB[0]))
            if m.has_bias:
                _call("tl_riscv_add_u8", NQ, T.address_of(Q[0]),
                      T.address_of(W[T.cast(b + o["qkv_bias"], "int64")]))
                _call("tl_riscv_add_u8", KVD, T.address_of(KB[0]),
                      T.address_of(W[T.cast(b + o["qkv_bias"] + NQ * 4, "int64")]))
                _call("tl_riscv_add_u8", KVD, T.address_of(VB[0]),
                      T.address_of(W[T.cast(b + o["qkv_bias"] + (NQ + KVD) * 4, "int64")]))
            for h in range(NH):
                if m.has_qk_norm:
                    _call("tl_riscv_head_norm", HD, T.address_of(Q[h * HD]),
                          T.address_of(W[T.cast(b + o["q_norm"], "int64")]), eps)
                _call("tl_riscv_rope", HD, T.address_of(Q[h * HD]),
                      T.address_of(CS[p, 0]), T.address_of(SN[p, 0]))
            for h in range(NKV):
                if m.has_qk_norm:
                    _call("tl_riscv_head_norm", HD, T.address_of(KB[h * HD]),
                          T.address_of(W[T.cast(b + o["k_norm"], "int64")]), eps)
                _call("tl_riscv_rope", HD, T.address_of(KB[h * HD]),
                      T.address_of(CS[p, 0]), T.address_of(SN[p, 0]))
            for h in range(NKV):
                _call("tl_riscv_copy", HD,
                      T.address_of(KC[((l * NKV + h) * maxseq + p) * HD]),
                      T.address_of(KB[h * HD]))
                _call("tl_riscv_copy", HD,
                      T.address_of(VC[((l * NKV + h) * maxseq + p) * HD]),
                      T.address_of(VB[h * HD]))
            _call("tl_riscv_attn_decode", NH, m.nkv_attn, HD, p, maxseq * HD,
                  scale, T.address_of(Q[0]),
                  T.address_of(KC[l * NKV * maxseq * HD]),
                  T.address_of(VC[l * NKV * maxseq * HD]),
                  T.address_of(ATT[0]))
            _call(GEMV, H, NQ,
                  T.address_of(W[T.cast(b + o["wo.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wo.s"], "int64")]),
                  T.address_of(ATT[0]), T.address_of(XB[0]))
            _call("tl_riscv_add", H, T.address_of(X[0]), T.address_of(XB[0]))

            _call("tl_riscv_rmsnorm", H, T.address_of(X[0]),
                  T.address_of(W[T.cast(b + o["rms_ffn"], "int64")]), eps,
                  T.address_of(XB[0]))
            _call(GEMV, IM, H,
                  T.address_of(W[T.cast(b + o["wgate.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wgate.s"], "int64")]),
                  T.address_of(XB[0]), T.address_of(GU[0]))
            _call(GEMV, IM, H,
                  T.address_of(W[T.cast(b + o["wup.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wup.s"], "int64")]),
                  T.address_of(XB[0]), T.address_of(GU[IM]))
            _call("tl_riscv_silu_mul", IM, T.address_of(GU[0]),
                  T.address_of(GU[IM]), T.address_of(GU[0]))
            _call(GEMV, H, IM,
                  T.address_of(W[T.cast(b + o["wdown.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wdown.s"], "int64")]),
                  T.address_of(GU[0]), T.address_of(XB[0]))
            _call("tl_riscv_add", H, T.address_of(X[0]), T.address_of(XB[0]))

    return layer


def build_layer_m8(m: Meta, maxseq: int, MR: int = 8):
    """Same layer, MR prompt tokens at a time (prefill only).

    Every linear layer reads its weights once for MR rows instead of once per
    token.  Attention stays per row so the causal order inside the chunk is
    preserved: row r's KV is written before row r attends, and it sees only
    positions <= its own.
    """
    H, NH, NKV, HD, IM = m.H, m.NH, m.NKV, m.HD, m.IM
    NQ, KVD = m.NQ, m.KV
    eps = m.eps
    o = m.layer_off
    scale = 1.0 / (HD ** 0.5)
    GEMV = _gemv_name(m, batched=True)
    # down_proj slices, computed outside the traced function: min()/comparisons
    # on traced expressions are illegal inside @T.prim_func.
    _KS = 1024
    DOWN_SLICES = [(si * _KS, min(_KS, IM - si * _KS))
                   for si in range((IM + _KS - 1) // _KS)]
    GEMV_WIDE = GEMV

    @T.prim_func
    def layer(
        W: T.Tensor((m.nbytes,), "uint8"),
        CS: T.Tensor((maxseq, HD // 2), "float32"),
        SN: T.Tensor((maxseq, HD // 2), "float32"),
        KC: T.Tensor((m.L * NKV * maxseq * HD,), "float32"),
        VC: T.Tensor((m.L * NKV * maxseq * HD,), "float32"),
        X: T.Tensor((MR * H,), "float32"),
        XB: T.Tensor((MR * H,), "float32"),
        Q: T.Tensor((MR * NQ,), "float32"),
        KB: T.Tensor((MR * KVD,), "float32"),
        VB: T.Tensor((MR * KVD,), "float32"),
        ATT: T.Tensor((MR * NQ,), "float32"),
        GUG: T.Tensor((MR * IM,), "float32"),
        GUU: T.Tensor((MR * IM,), "float32"),
        POS: T.Tensor((1,), "int32"),
        BASE: T.Tensor((1,), "int64"),
        LI: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            b = BASE[0]
            l = LI[0]
            for mm in range(MR):
                _call("tl_riscv_rmsnorm", H, T.address_of(X[mm * H]),
                      T.address_of(W[T.cast(b + o["rms_attn"], "int64")]), eps,
                      T.address_of(XB[mm * H]))
            _call(GEMV, NQ, H, T.address_of(W[T.cast(b + o["wq.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wq.s"], "int64")]),
                  T.address_of(XB[0]), T.address_of(Q[0]))
            _call(GEMV, KVD, H, T.address_of(W[T.cast(b + o["wk.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wk.s"], "int64")]),
                  T.address_of(XB[0]), T.address_of(KB[0]))
            _call(GEMV, KVD, H, T.address_of(W[T.cast(b + o["wv.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wv.s"], "int64")]),
                  T.address_of(XB[0]), T.address_of(VB[0]))
            if m.has_bias:
                for mm in range(MR):
                    _call("tl_riscv_add_u8", NQ, T.address_of(Q[mm * NQ]),
                          T.address_of(W[T.cast(b + o["qkv_bias"], "int64")]))
                    _call("tl_riscv_add_u8", KVD, T.address_of(KB[mm * KVD]),
                          T.address_of(W[T.cast(b + o["qkv_bias"] + NQ * 4, "int64")]))
                    _call("tl_riscv_add_u8", KVD, T.address_of(VB[mm * KVD]),
                          T.address_of(W[T.cast(b + o["qkv_bias"] + (NQ + KVD) * 4, "int64")]))
            for mm in range(MR):
                for h in range(NH):
                    if m.has_qk_norm:
                        _call("tl_riscv_head_norm", HD,
                              T.address_of(Q[mm * NQ + h * HD]),
                              T.address_of(W[T.cast(b + o["q_norm"], "int64")]), eps)
                    _call("tl_riscv_rope", HD, T.address_of(Q[mm * NQ + h * HD]),
                          T.address_of(CS[POS[0] + mm, 0]),
                          T.address_of(SN[POS[0] + mm, 0]))
                for h in range(NKV):
                    if m.has_qk_norm:
                        _call("tl_riscv_head_norm", HD,
                              T.address_of(KB[mm * KVD + h * HD]),
                              T.address_of(W[T.cast(b + o["k_norm"], "int64")]), eps)
                    _call("tl_riscv_rope", HD, T.address_of(KB[mm * KVD + h * HD]),
                          T.address_of(CS[POS[0] + mm, 0]),
                          T.address_of(SN[POS[0] + mm, 0]))
            for mm in range(MR):
                for h in range(NKV):
                    _call("tl_riscv_copy", HD,
                          T.address_of(KC[((l * NKV + h) * maxseq + POS[0] + mm) * HD]),
                          T.address_of(KB[mm * KVD + h * HD]))
                    _call("tl_riscv_copy", HD,
                          T.address_of(VC[((l * NKV + h) * maxseq + POS[0] + mm) * HD]),
                          T.address_of(VB[mm * KVD + h * HD]))
                _call("tl_riscv_attn_decode", NH, m.nkv_attn, HD, POS[0] + mm,
                      maxseq * HD, scale, T.address_of(Q[mm * NQ]),
                      T.address_of(KC[l * NKV * maxseq * HD]),
                      T.address_of(VC[l * NKV * maxseq * HD]),
                      T.address_of(ATT[mm * NQ]))
            _call(GEMV, H, NQ, T.address_of(W[T.cast(b + o["wo.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wo.s"], "int64")]),
                  T.address_of(ATT[0]), T.address_of(XB[0]))
            for mm in range(MR):
                _call("tl_riscv_add", H, T.address_of(X[mm * H]),
                      T.address_of(XB[mm * H]))
                _call("tl_riscv_rmsnorm", H, T.address_of(X[mm * H]),
                      T.address_of(W[T.cast(b + o["rms_ffn"], "int64")]), eps,
                      T.address_of(XB[mm * H]))
            _call(GEMV, IM, H, T.address_of(W[T.cast(b + o["wgate.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wgate.s"], "int64")]),
                  T.address_of(XB[0]), T.address_of(GUG[0]))
            _call(GEMV, IM, H, T.address_of(W[T.cast(b + o["wup.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wup.s"], "int64")]),
                  T.address_of(XB[0]), T.address_of(GUU[0]))
            for mm in range(MR):
                _call("tl_riscv_silu_mul", IM, T.address_of(GUG[mm * IM]),
                      T.address_of(GUU[mm * IM]), T.address_of(GUG[mm * IM]))
            # down_proj is the only K=IM matrix.  Calling it in 1024-element
            # slices keeps each 8x1024 activation tile in L1 (32 KB) and costs
            # one OpenMP region per slice; the whole-K call re-reads a 96 KB
            # (0.6B) to 393 KB (8B) tile per output row and runs at ~1.1 GB/s.
            for si, (qoff, klen) in enumerate(DOWN_SLICES):
                _call("tl_riscv_gemv_q8_m8_slice", H, klen,
                      T.address_of(W[T.cast(b + o["wdown.q"] + qoff, "int64")]),
                      T.address_of(W[T.cast(b + o["wdown.s"] + (qoff // 32) * 2, "int64")]),
                      T.address_of(GUG[qoff]), T.address_of(XB[0]),
                      0 if si == 0 else 1, IM, IM)
            for mm in range(MR):
                _call("tl_riscv_add", H, T.address_of(X[mm * H]),
                      T.address_of(XB[mm * H]))

    return layer


def build_head(m: Meta):
    H, V = m.H, m.V
    GEMV = _gemv_name(m)

    @T.prim_func
    def head(
        W: T.Tensor((m.nbytes,), "uint8"),
        X: T.Tensor((H,), "float32"),
        XB: T.Tensor((H,), "float32"),
        OUT: T.Tensor((1,), "int32"),
        LOGITS: T.Tensor((V,), "float32"),
    ):
        with T.Kernel(1):
            _call("tl_riscv_rmsnorm", H, T.address_of(X[0]),
                  T.address_of(W[T.cast(m.final_norm, "int64")]), m.eps,
                  T.address_of(XB[0]))
            _call(GEMV, V, H,
                  T.address_of(W[T.cast(m.lm_q, "int64")]),
                  T.address_of(W[T.cast(m.lm_s, "int64")]),
                  T.address_of(XB[0]), T.address_of(LOGITS[0]))
            _call("tl_riscv_argmax", V, T.address_of(LOGITS[0]),
                  T.address_of(OUT[0]))

    return head


class Engine:
    def __init__(self, blob_path: str, maxseq: int = 512):
        meta = json.load(open(blob_path + ".json"))
        kw = {k: meta[k] for k in Meta.__dataclass_fields__ if k in meta}
        self.m = Meta(**kw)
        if not self.m.nkv_attn:
            self.m.nkv_attn = self.m.NKV
        self.maxseq = maxseq
        if self.m.layout == "ime2q4":
            # The A100 IME2 Q4 GEMV lives in libime2bridge.so (vendor spert
            # kernels).  Load it RTLD_GLOBAL before the TileLang module does,
            # the same contract the OpenBLAS path uses.
            import ctypes
            cands = [os.environ.get("IME2_BRIDGE", ""),
                     os.path.join(os.path.dirname(__file__), "../../../build_riscv",
                                  "libime2bridge.so"),
                     "/usr/local/lib/libime2bridge.so"]
            so = next((c for c in cands if c and os.path.exists(c)), None)
            if not so:
                raise RuntimeError(
                    "ime2q4 blob needs libime2bridge.so; set IME2_BRIDGE")
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            print(f"[engine] IME2 Q4 decode path ({so})", flush=True)
        self.w = np.memmap(blob_path, dtype=np.uint8, mode="c")
        assert self.w.size == self.m.nbytes, (self.w.size, self.m.nbytes)
        m = self.m
        self.kc = np.zeros(m.L * m.NKV * maxseq * m.HD, dtype=np.float32)
        self.vc = np.zeros_like(self.kc)
        self.x = np.zeros(m.H, dtype=np.float32)
        self.xb = np.zeros(m.H, dtype=np.float32)
        self.q = np.zeros(m.NQ, dtype=np.float32)
        self.kb = np.zeros(m.KV, dtype=np.float32)
        self.vb = np.zeros(m.KV, dtype=np.float32)
        self.att = np.zeros(m.NQ, dtype=np.float32)
        self.gu = np.zeros(2 * m.IM, dtype=np.float32)
        self.out = np.zeros(1, dtype=np.int32)
        self.logits = np.zeros(m.V, dtype=np.float32)
        self.pos = np.zeros(1, dtype=np.int32)
        self.base = np.zeros(1, dtype=np.int64)
        self.li = np.zeros(1, dtype=np.int32)
        self.ids = np.zeros(1, dtype=np.int32)
        # prefill batch buffers (MR=8 rows)
        self.MR = 8
        self.x8 = np.zeros(self.MR * m.H, dtype=np.float32)
        self.xb8 = np.zeros(self.MR * m.H, dtype=np.float32)
        self.q8b = np.zeros(self.MR * m.NQ, dtype=np.float32)
        self.kb8 = np.zeros(self.MR * m.KV, dtype=np.float32)
        self.vb8 = np.zeros(self.MR * m.KV, dtype=np.float32)
        self.att8 = np.zeros(self.MR * m.NQ, dtype=np.float32)
        self.gug8 = np.zeros(self.MR * m.IM, dtype=np.float32)
        self.guu8 = np.zeros(self.MR * m.IM, dtype=np.float32)
        self.posn = 0
        self.last_prefill_s = 0.0
        self.last_decode_s = 0.0
        half = m.HD // 2
        inv = m.theta ** (-2.0 * np.arange(half, dtype=np.float32) / m.HD)
        ang = np.arange(maxseq, dtype=np.float32)[:, None] * inv[None, :]
        self.cs = np.cos(ang).astype(np.float32)
        self.sn = np.sin(ang).astype(np.float32)

        t0 = time.time()
        self.f_step = None
        self.f_layer = None
        with tvm.target.Target(RISCV_TARGET):
            # Prefer the whole-model program: one FFI call per token instead of
            # L+1, worth ~7 ms/token on this board.  It only lowers for some
            # shape combinations (see module docstring), so fall back to the
            # per-layer program when TVM rejects it.
            try:
                self.f_step = tilelang.compile(build_step(m, maxseq),
                                               out_idx=None,
                                               execution_backend="tvm_ffi")
            except Exception as exc:  # noqa: BLE001 - compiler bug, not ours
                print(f"[engine] whole-model lowering failed ({exc}); "
                      f"using per-layer programs", flush=True)
                self.f_layer = tilelang.compile(build_layer(m, maxseq),
                                                out_idx=None,
                                                execution_backend="tvm_ffi")
            # the head is also used by the batched prefill path, so compile it
            # even when the whole-model program exists.
            self.f_head = tilelang.compile(build_head(m), out_idx=None,
                                           execution_backend="tvm_ffi")
            self.f_layer_m8 = None
            # Measured: the batched layer is 1.4x faster at H=1024 (0.6B) but
            # 5.5x SLOWER at H=4096 (8B, 3.0 s vs 0.54 s per token) -- the
            # per-row work and the 12 down_proj slices per layer dominate once
            # the weight matrices grow.  Gate it on the hidden size instead of
            # shipping a regression.
            if m.H > 2048:
                self.build_s = time.time() - t0
                print(f"[engine] batched prefill off for H={m.H} "
                      f"(measured slower above H=2048)", flush=True)
            try:
                if m.layout != "q8planar":
                    raise RuntimeError("batched prefill needs the RVV Q8 kernel")
                if m.H > 2048:
                    raise RuntimeError("batched prefill disabled for large models")
                if os.environ.get("TL_NO_BATCH"):
                    raise RuntimeError("batched prefill disabled by TL_NO_BATCH")
                self.f_layer_m8 = tilelang.compile(build_layer_m8(m, maxseq),
                                                   out_idx=None,
                                                   execution_backend="tvm_ffi")
            except Exception as exc:  # noqa: BLE001
                print(f"[engine] batched prefill unavailable ({exc})", flush=True)
        self.build_s = time.time() - t0

    def _embed(self, tok: int):
        m = self.m
        self.ids[0] = tok
        row = np.frombuffer(
            self.w[m.embed + tok * m.H * 2: m.embed + (tok + 1) * m.H * 2],
            dtype=np.float16)
        self.x[:] = row

    def _embed_row(self, dst: np.ndarray, row: int, tok: int):
        m = self.m
        dst[row * m.H:(row + 1) * m.H] = np.frombuffer(
            self.w[m.embed + tok * m.H * 2: m.embed + (tok + 1) * m.H * 2],
            dtype=np.float16)

    def _prefill_chunk(self, chunk: list[int], base_pos: int) -> int:
        m = self.m
        for i, t in enumerate(chunk):
            self._embed_row(self.x8, i, t)
        self.pos[0] = base_pos
        for l in range(m.L):
            self.base[0] = m.layer_base + l * m.layer_stride
            self.li[0] = l
            self.f_layer_m8(self.w, self.cs, self.sn, self.kc, self.vc, self.x8,
                            self.xb8, self.q8b, self.kb8, self.vb8, self.att8,
                            self.gug8, self.guu8, self.pos, self.base, self.li)
        last = self.MR - 1
        self.x[:] = self.x8[last * m.H:(last + 1) * m.H]
        self.f_head(self.w, self.x, self.xb, self.out, self.logits)
        return int(self.out[0])

    def step(self, tok: int, pos: int) -> int:
        self.pos[0] = pos
        m = self.m
        self._embed(tok)
        if self.f_step is not None:
            self.f_step(self.w, self.cs, self.sn, self.kc, self.vc, self.x,
                        self.xb, self.q, self.kb, self.vb, self.att, self.gu,
                        self.ids, self.pos, self.out, self.logits)
            return int(self.out[0])
        for l in range(m.L):
            self.base[0] = m.layer_base + l * m.layer_stride
            self.li[0] = l
            self.f_layer(self.w, self.cs, self.sn, self.kc, self.vc, self.x,
                         self.xb, self.q, self.kb, self.vb, self.att, self.gu,
                         self.pos, self.base, self.li)
        self.f_head(self.w, self.x, self.xb, self.out, self.logits)
        return int(self.out[0])

    def reset(self):
        self.posn = 0
        self.kc[:] = 0
        self.vc[:] = 0

    def generate(self, prompt_ids: list[int], n_new: int,
                 eos: tuple[int, ...] = ()) -> tuple[list[int], float]:
        self.reset()
        out = []
        t0 = time.time()
        tok = prompt_ids[0]
        pos = 0
        if self.f_layer_m8 is not None:
            while pos + self.MR <= len(prompt_ids):
                tok = self._prefill_chunk(prompt_ids[pos:pos + self.MR], pos)
                pos += self.MR
        for i in range(pos, len(prompt_ids)):
            tok = self.step(prompt_ids[i], i)
        self.posn = len(prompt_ids)
        self.last_prefill_s = time.time() - t0
        t1 = time.time()
        for _ in range(n_new):
            out.append(tok)
            if tok in eos:
                break
            tok = self.step(tok, self.posn)
            self.posn += 1
        self.last_decode_s = time.time() - t1
        return out, time.time() - t0


def load_tokenizer(path: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)
