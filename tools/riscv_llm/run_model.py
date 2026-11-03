"""Run a model end to end on the TileLang RISC-V backend.

    python tools/riscv_llm/run_model.py <blob.bin> <tokenizer.json> \
        --prompt "..." --n 32 [--maxseq 512] [--threads 8]

Prints the generated text plus prefill/decode throughput.  Thread count is
taken from OMP_NUM_THREADS at process start (OpenMP caches the pool).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from tilelang.riscv.llm.engine import Engine, load_tokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    ap.add_argument("tokenizer")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--maxseq", type=int, default=512)
    ap.add_argument("--raw", action="store_true",
                    help="no chat template, plain completion")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    eng = Engine(args.blob, maxseq=args.maxseq)
    m = eng.m
    print(f"model H={m.H} L={m.L} NH={m.NH} NKV={m.NKV} HD={m.HD} IM={m.IM} "
          f"V={m.V} qk_norm={m.has_qk_norm} bias={m.has_bias} "
          f"weights={m.nbytes/1e9:.2f}GB build={eng.build_s:.1f}s", flush=True)

    text = args.prompt
    if not args.raw:
        text = (f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    ids = tok.encode(text).ids
    print(f"prompt tokens: {len(ids)}", flush=True)

    t0 = time.time()
    out, dt = eng.generate(ids, args.n, eos=(151645, 151643))
    gen = tok.decode(out)
    n_dec = max(1, len(out))
    print(f"--- output ({len(out)} tokens, {dt:.2f}s, "
          f"{n_dec/dt:.2f} tok/s incl. prefill) ---")
    print(f"    prefill {len(ids)} tok, decode {n_dec} tok: "
          f"{eng.last_prefill_s*1e3:.0f} ms + {eng.last_decode_s*1e3:.0f} ms"
          f" -> {n_dec/max(1e-9, eng.last_decode_s):.2f} tok/s decode",
          flush=True)
    print(gen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
