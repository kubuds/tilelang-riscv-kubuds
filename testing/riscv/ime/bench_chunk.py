"""Time one prefill chunk in place, so MR=4 vs MR=8 is an apples-to-apples A/B.

No prompt loop, no tokenizer warm-up variance: 28 layer programs over a fixed
chunk, repeated, min of reps.  Run twice (TL_MR=4, TL_MR=8) via bench_chunk.sh.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/work/kubuds/hermes/tilelang_riscv")
from tilelang.riscv.llm.engine import Engine, load_tokenizer  # noqa: E402

blob = "/work/kubuds/hermes/models/tl_riscv/qwen3_06b_q4_ime2.bin"
tokp = "/work/kubuds/hermes/models/qwen3_06b_hf/tokenizer.json"

tok = load_tokenizer(tokp)
ids = tok.encode("The capital of France is Paris, and the capital of Italy is Rome, and Spain is").ids
eng = Engine(blob, maxseq=128)
MR = eng.MR
chunk = (ids * 4)[:MR]
m = eng.m

# warm up
for _ in range(2):
    eng._prefill_chunk(chunk, 0)

best = 1e9
for _ in range(7):
    t0 = time.perf_counter()
    for _ in range(5):
        eng._prefill_chunk(chunk, 0)
    best = min(best, (time.perf_counter() - t0) / 5)

print(f"MR={MR}  chunk {best*1e3:.1f} ms  -> {best*1e3/MR:.2f} ms/token "
      f"({MR/best:.1f} tok/s)")
