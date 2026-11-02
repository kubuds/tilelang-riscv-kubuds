"""GQA attention (single query, seq=128) as three TileLang kernels on X100."""
import time
import torch
import tilelang
import tilelang.cpu.language as T

NH, NKV, HD, SEQ = 16, 8, 128, 128
SCALE = 1.0 / (HD ** 0.5)
DT = "float32"
TARGET = {"kind": "c"}


@tilelang.jit(target=TARGET, out_idx=[2], execution_backend="cython")
def score_kernel():
    @T.prim_func
    def main(Q: T.Tensor((NH, HD), DT), K: T.Tensor((NKV, SEQ, HD), DT), S: T.Tensor((NH, SEQ), DT)):
        with T.Kernel(NH) as h:
            for t in T.Parallel(SEQ):
                acc = T.alloc_local((1,), DT)
                acc[0] = 0.0
                for d in T.serial(HD):
                    acc[0] = acc[0] + Q[h, d] * K[h // 2, t, d]
                S[h, t] = acc[0] * SCALE
    return main


@tilelang.jit(target=TARGET, out_idx=[1])
def softmax_kernel():
    @T.prim_func
    def main(S: T.Tensor((NH, SEQ), DT), P: T.Tensor((NH, SEQ), DT)):
        with T.Kernel(NH) as h:
            mx = T.alloc_local((1,), DT)
            sm = T.alloc_local((1,), DT)
            mx[0] = -1e30
            for t in T.serial(SEQ):
                if S[h, t] > mx[0]:
                    mx[0] = S[h, t]
            sm[0] = 0.0
            for t in T.serial(SEQ):
                P[h, t] = T.exp(S[h, t] - mx[0])
                sm[0] = sm[0] + P[h, t]
            for t in T.serial(SEQ):
                P[h, t] = P[h, t] / sm[0]
    return main


@tilelang.jit(target=TARGET, out_idx=[2], execution_backend="cython")
def av_kernel():
    @T.prim_func
    def main(P: T.Tensor((NH, SEQ), DT), V: T.Tensor((NKV, SEQ, HD), DT), O: T.Tensor((NH, HD), DT)):
        with T.Kernel(NH) as h:
            for d in T.Parallel(HD):
                acc = T.alloc_local((1,), DT)
                acc[0] = 0.0
                for t in T.serial(SEQ):
                    acc[0] = acc[0] + P[h, t] * V[h // 2, t, d]
                O[h, d] = acc[0]
    return main


torch.manual_seed(0)
q = torch.randn(NH, HD)
k = torch.randn(NKV, SEQ, HD)
v = torch.randn(NKV, SEQ, HD)
ks, ksm, ka = score_kernel(), softmax_kernel(), av_kernel()

s = ks(q, k)
p = ksm(s)
o = ka(p, v)
# reference
sref = torch.stack([q[h] @ k[h // 2].T for h in range(NH)]) * SCALE
pref = torch.softmax(sref, dim=-1)
oref = torch.stack([pref[h] @ v[h // 2] for h in range(NH)])
print("score err", (s - sref).abs().max().item())
print("prob  err", (p - pref).abs().max().item())
print("out   err", (o - oref).abs().max().item())


def fwd():
    s = ks(q, k)
    p = ksm(s)
    return ka(p, v)


def bench(fn, *args, it=50):
    for _ in range(3):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(it):
        fn(*args)
    return (time.perf_counter() - t0) / it * 1e3
print(f"score: {bench(ks,q,k):.3f} ms  softmax: {bench(ksm,s):.3f} ms  av: {bench(ka,p,v):.3f} ms")
