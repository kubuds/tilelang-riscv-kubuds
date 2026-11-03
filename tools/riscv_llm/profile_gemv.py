"""Where does a decode step's time go: weight streaming vs the rest.

Builds the same layer program as engine.py but with only the seven GEMVs (no
RMSNorm, RoPE, attention, SiLU, KV copy), runs it 28 times plus the lm_head,
and compares with the full engine step.  The difference is the non-GEMV cost.
"""


import json
import time

import numpy as np
import tilelang
import tilelang.language as T
from tilelang import tvm

from tilelang.riscv.llm.engine import Meta, RISCV_TARGET

BLOB = "/work/kubuds/hermes/models/tl_riscv/qwen3_06b_int8.bin"


def build_gemvs(m: Meta):
    H, NH, NKV, HD, IM = m.H, m.NH, m.NKV, m.HD, m.IM
    NQ, KVD = m.NQ, m.KV
    o = m.layer_off

    @T.prim_func
    def gemvs(
        W: T.Tensor((m.nbytes,), "uint8"),
        X: T.Tensor((H,), "float32"),
        Q: T.Tensor((NQ,), "float32"),
        KB: T.Tensor((KVD,), "float32"),
        VB: T.Tensor((KVD,), "float32"),
        ATT: T.Tensor((NQ,), "float32"),
        GU: T.Tensor((2 * IM,), "float32"),
        XB: T.Tensor((H,), "float32"),
        BASE: T.Tensor((1,), "int64"),
    ):
        with T.Kernel(1):
            b = BASE[0]
            T.call_extern("int32", "tl_riscv_gemv_q8", NQ, H,
                          T.address_of(W[T.cast(b + o["wq.q"], "int64")]),
                          T.address_of(W[T.cast(b + o["wq.s"], "int64")]),
                          T.address_of(X[0]), T.address_of(Q[0]))
            T.call_extern("int32", "tl_riscv_gemv_q8", KVD, H,
                          T.address_of(W[T.cast(b + o["wk.q"], "int64")]),
                          T.address_of(W[T.cast(b + o["wk.s"], "int64")]),
                          T.address_of(X[0]), T.address_of(KB[0]))
            T.call_extern("int32", "tl_riscv_gemv_q8", KVD, H,
                          T.address_of(W[T.cast(b + o["wv.q"], "int64")]),
                          T.address_of(W[T.cast(b + o["wv.s"], "int64")]),
                          T.address_of(X[0]), T.address_of(VB[0]))
            T.call_extern("int32", "tl_riscv_gemv_q8", H, NQ,
                          T.address_of(W[T.cast(b + o["wo.q"], "int64")]),
                          T.address_of(W[T.cast(b + o["wo.s"], "int64")]),
                          T.address_of(ATT[0]), T.address_of(XB[0]))
            T.call_extern("int32", "tl_riscv_gemv_q8", IM, H,
                          T.address_of(W[T.cast(b + o["wgate.q"], "int64")]),
                          T.address_of(W[T.cast(b + o["wgate.s"], "int64")]),
                          T.address_of(XB[0]), T.address_of(GU[0]))
            T.call_extern("int32", "tl_riscv_gemv_q8", IM, H,
                          T.address_of(W[T.cast(b + o["wup.q"], "int64")]),
                          T.address_of(W[T.cast(b + o["wup.s"], "int64")]),
                          T.address_of(XB[0]), T.address_of(GU[IM]))
            T.call_extern("int32", "tl_riscv_gemv_q8", H, IM,
                          T.address_of(W[T.cast(b + o["wdown.q"], "int64")]),
                          T.address_of(W[T.cast(b + o["wdown.s"], "int64")]),
                          T.address_of(GU[0]), T.address_of(XB[0]))

    return gemvs


def main() -> int:
    meta = json.load(open(BLOB + ".json"))
    m = Meta(**{k: meta[k] for k in Meta.__dataclass_fields__ if k in meta})
    if not m.nkv_attn:
        m.nkv_attn = m.NKV
    w = np.memmap(BLOB, dtype=np.uint8, mode="c")
    x = np.zeros(m.H, np.float32)
    xb = np.zeros(m.H, np.float32)
    q = np.zeros(m.NQ, np.float32)
    kb = np.zeros(m.KV, np.float32)
    vb = np.zeros(m.KV, np.float32)
    att = np.zeros(m.NQ, np.float32)
    gu = np.zeros(2 * m.IM, np.float32)
    base = np.zeros(1, np.int64)

    t0 = time.time()
    with tvm.target.Target(RISCV_TARGET):
        f = tilelang.compile(build_gemvs(m), out_idx=None,
                             execution_backend="tvm_ffi")
    print(f"build {time.time()-t0:.1f}s", flush=True)

    def run():
        for l in range(m.L):
            base[0] = m.layer_base + l * m.layer_stride
            f(w, x, q, kb, vb, att, gu, xb, base)

    run()
    best = 1e9
    for _ in range(3):
        t = time.time()
        run()
        best = min(best, time.time() - t)
    per = best / m.L
    print(f"layer GEMVs: {per*1e3:.2f} ms/layer  x{m.L} = {best*1e3:.1f} ms/token")
    print(f"  {m.layer_stride/1e6:.1f} MB per layer -> {m.L*m.layer_stride/best/1e9:.1f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
