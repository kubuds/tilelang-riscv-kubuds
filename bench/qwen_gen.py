"""Real Qwen3-0.6B greedy generation through the single-launch graph."""
import time
import numpy as np

exec(open("/work/mahiru/src/tl/tilelang-0.1.15/bench/qwen_real.py").read())

EMB = data
K = 1024


def embed(t):
    r = EMB[t * 576:(t + 1) * 576]
    q = np.frombuffer(r, dtype=np.uint8).reshape(K // 32, 18)
    d = q[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    qs = q[:, 2:]
    lo = ((qs & 0xF).astype(np.int16) - 8).astype(np.float32)
    hi = ((qs >> 4).astype(np.int16) - 8).astype(np.float32)
    w = np.empty((K // 32, 32), dtype=np.float32)
    w[:, :16] = lo * d
    w[:, 16:] = hi * d
    return w.reshape(K)


def toks_text(ids):
    s = "".join(tokens[i] for i in ids)
    return s.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")


prompt = [785, 6722, 315, 9625, 374]
KC[:] = 0
VC[:] = 0
for pos, t in enumerate(prompt):
    xv[:] = embed(t)
    rc = lib.qwen_graph_fwd(arr, pos, MAXT)
    assert rc == 0
first = int(logits[VOCAB])
print("after prompt, next token", first, repr(tokens[first]))

gen = [first]
t0 = time.perf_counter()
for i in range(16):
    xv[:] = embed(gen[-1])
    lib.qwen_graph_fwd(arr, len(prompt) + i, MAXT)
    gen.append(int(logits[VOCAB]))
dt = time.perf_counter() - t0
print("generated:", repr(toks_text(gen)))
print(f"decode {len(gen)-1} tokens in {dt*1e3:.1f} ms -> {(len(gen)-1)/dt:.1f} tok/s")
