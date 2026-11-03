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

    @property
    def NQ(self):
        return self.NH * self.HD

    @property
    def KV(self):
        return self.NKV * self.HD


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
                _call("tl_riscv_gemv_q8", NQ, H, T.address_of(W[T.cast(b + o["wq.q"], 'int64')]),
                      T.address_of(W[T.cast(b + o["wq.s"], 'int64')]), T.address_of(XB[0]),
                      T.address_of(Q[0]))
                _call("tl_riscv_gemv_q8", KVD, H, T.address_of(W[T.cast(b + o["wk.q"], 'int64')]),
                      T.address_of(W[T.cast(b + o["wk.s"], 'int64')]), T.address_of(XB[0]),
                      T.address_of(KB[0]))
                _call("tl_riscv_gemv_q8", KVD, H, T.address_of(W[T.cast(b + o["wv.q"], 'int64')]),
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
                _call("tl_riscv_gemv_q8", H, NQ, T.address_of(W[T.cast(b + o["wo.q"], 'int64')]),
                      T.address_of(W[T.cast(b + o["wo.s"], 'int64')]), T.address_of(ATT[0]),
                      T.address_of(XB[0]))
                _call("tl_riscv_add", H, T.address_of(X[0]),
                      T.address_of(XB[0]))

                _call("tl_riscv_rmsnorm", H, T.address_of(X[0]),
                      T.address_of(W[T.cast(b + o["rms_ffn"], 'int64')]), eps, T.address_of(XB[0]))
                _call("tl_riscv_gemv_q8", IM, H, T.address_of(W[T.cast(b + o["wgate.q"], 'int64')]),
                      T.address_of(W[T.cast(b + o["wgate.s"], 'int64')]), T.address_of(XB[0]),
                      T.address_of(GU[0]))
                _call("tl_riscv_gemv_q8", IM, H, T.address_of(W[T.cast(b + o["wup.q"], 'int64')]),
                      T.address_of(W[T.cast(b + o["wup.s"], 'int64')]), T.address_of(XB[0]),
                      T.address_of(GU[IM]))
                _call("tl_riscv_silu_mul", IM, T.address_of(GU[0]),
                      T.address_of(GU[IM]), T.address_of(GU[0]))
                _call("tl_riscv_gemv_q8", H, IM, T.address_of(W[T.cast(b + o["wdown.q"], 'int64')]),
                      T.address_of(W[T.cast(b + o["wdown.s"], 'int64')]), T.address_of(GU[0]),
                      T.address_of(XB[0]))
                _call("tl_riscv_add", H, T.address_of(X[0]),
                      T.address_of(XB[0]))

            _call("tl_riscv_rmsnorm", H, T.address_of(X[0]),
                  T.address_of(W[T.cast(m.final_norm, 'int64')]), eps, T.address_of(XB[0]))
            _call("tl_riscv_gemv_q8", V, H, T.address_of(W[T.cast(m.lm_q, 'int64')]),
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
            _call("tl_riscv_gemv_q8", NQ, H,
                  T.address_of(W[T.cast(b + o["wq.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wq.s"], "int64")]),
                  T.address_of(XB[0]), T.address_of(Q[0]))
            _call("tl_riscv_gemv_q8", KVD, H,
                  T.address_of(W[T.cast(b + o["wk.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wk.s"], "int64")]),
                  T.address_of(XB[0]), T.address_of(KB[0]))
            _call("tl_riscv_gemv_q8", KVD, H,
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
            _call("tl_riscv_gemv_q8", H, NQ,
                  T.address_of(W[T.cast(b + o["wo.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wo.s"], "int64")]),
                  T.address_of(ATT[0]), T.address_of(XB[0]))
            _call("tl_riscv_add", H, T.address_of(X[0]), T.address_of(XB[0]))

            _call("tl_riscv_rmsnorm", H, T.address_of(X[0]),
                  T.address_of(W[T.cast(b + o["rms_ffn"], "int64")]), eps,
                  T.address_of(XB[0]))
            _call("tl_riscv_gemv_q8", IM, H,
                  T.address_of(W[T.cast(b + o["wgate.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wgate.s"], "int64")]),
                  T.address_of(XB[0]), T.address_of(GU[0]))
            _call("tl_riscv_gemv_q8", IM, H,
                  T.address_of(W[T.cast(b + o["wup.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wup.s"], "int64")]),
                  T.address_of(XB[0]), T.address_of(GU[IM]))
            _call("tl_riscv_silu_mul", IM, T.address_of(GU[0]),
                  T.address_of(GU[IM]), T.address_of(GU[0]))
            _call("tl_riscv_gemv_q8", H, IM,
                  T.address_of(W[T.cast(b + o["wdown.q"], "int64")]),
                  T.address_of(W[T.cast(b + o["wdown.s"], "int64")]),
                  T.address_of(GU[0]), T.address_of(XB[0]))
            _call("tl_riscv_add", H, T.address_of(X[0]), T.address_of(XB[0]))

    return layer


def build_head(m: Meta):
    H, V = m.H, m.V

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
            _call("tl_riscv_gemv_q8", V, H,
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
                self.f_head = tilelang.compile(build_head(m), out_idx=None,
                                               execution_backend="tvm_ffi")
        self.build_s = time.time() - t0

    def _embed(self, tok: int):
        m = self.m
        self.ids[0] = tok
        row = np.frombuffer(
            self.w[m.embed + tok * m.H * 2: m.embed + (tok + 1) * m.H * 2],
            dtype=np.float16)
        self.x[:] = row

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
        for i, t in enumerate(prompt_ids):
            tok = self.step(t, i)
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
