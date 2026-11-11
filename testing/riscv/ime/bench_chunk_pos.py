"""Chunk cost vs context position: the slope is the attention term.

Everything else in a prefill chunk (7 GEMMs, rmsnorm, silu, KV copy) is
independent of how many positions came before; only attention grows with p.
So timing the same chunk at several base positions separates the two.
"""
import os
import sys
import time

sys.path.insert(0, "/work/kubuds/hermes/tilelang_riscv")
from tilelang.riscv.llm.engine import Engine, load_tokenizer  # noqa: E402

blob = "/work/kubuds/hermes/models/tl_riscv/qwen3_06b_q4_ime2.bin"
tokp = "/work/kubuds/hermes/models/qwen3_06b_hf/tokenizer.json"

tok = load_tokenizer(tokp)
ids = tok.encode("The capital of France is Paris, and the capital of Italy is Rome, and Spain is").ids
eng = Engine(blob, maxseq=512)
mr = int(os.environ.get("TL_MR", "16"))
fn = eng.pref_plans[mr]
chunk = (ids * 8)[:mr]

def run(pos):
    for _ in range(2):
        eng._prefill_chunk(chunk, pos, fn, mr)
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(3):
            eng._prefill_chunk(chunk, pos, fn, mr)
        best = min(best, (time.perf_counter() - t0) / 3)
    return best * 1e3

pts = [(p, run(p)) for p in (0, 32, 64, 128, 256)]
for p, ms in pts:
    print(f"pos {p:4d}  chunk {ms:6.1f} ms  {ms/mr:5.2f} ms/token")
(p0, t0), (p1, t1) = pts[0], pts[-1]
print(f"attention slope: {(t1-t0)/(p1-p0)*mr:.3f} ms per (chunk, position)")
print(f"at pos=128 attention = {(t1-t0)/(p1-p0)*128:.1f} ms/chunk of {t1:.1f}")
