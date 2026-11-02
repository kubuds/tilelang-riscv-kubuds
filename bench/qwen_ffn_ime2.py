"""Qwen3-0.6B FFN decode path with IME2 Q4_0 GEMV inside TileLang."""
import ctypes, time
import numpy as np
import torch
import tilelang
import tilelang.language as T

ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so", mode=ctypes.RTLD_GLOBAL)
lib = ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so")
lib.ime2_repack_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.ime2_gemv_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

HID, FFN = 1024, 3072
DT = "float32"
TARGET = {"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu",
          "mcpu": "generic-rv64", "mattr": ["+m", "+a", "+f", "+d", "+v"]}
rng = np.random.default_rng(0)


def make_q4(N, K):
    nb = K // 32
    d = rng.uniform(0.001, 0.02, size=(N, nb)).astype(np.float16)
    qs = rng.integers(0, 256, size=(N, nb, 16), dtype=np.uint8)
    raw = np.empty((N, nb, 18), dtype=np.uint8)
    raw[:, :, :2] = d.view(np.uint8).reshape(N, nb, 2)
    raw[:, :, 2:] = qs
    raw = raw.tobytes()
    lo = ((qs & 0x0F).astype(np.int16) - 8).astype(np.float32)
    hi = ((qs >> 4).astype(np.int16) - 8).astype(np.float32)
    w = np.empty((N, nb, 32), dtype=np.float32)
    w[:, :, :16] = lo * d.astype(np.float32)[:, :, None]
    w[:, :, 16:] = hi * d.astype(np.float32)[:, :, None]
    return raw, w.reshape(N, K)


def repack(raw, N, K):
    pb = (N // 32) * (K // 32) * 576
    packed = np.zeros(pb, dtype=np.uint8)
    r = lib.ime2_repack_q4(ctypes.c_char_p(raw), packed.ctypes.data, N, K)
    assert r == 0, r
    return packed


@tilelang.jit(target=TARGET, out_idx=[2])
def gemv(n, k):
    pb = (n // 32) * (k // 32) * 576
    @T.prim_func
    def main(P: T.Tensor((pb,), "uint8"), X: T.Tensor((k,), DT), Y: T.Tensor((n,), DT)):
        with T.Kernel(1, threads=1) as bx:
            T.call_extern("int", "ime2_gemv_q4", T.address_of(P[0]), T.address_of(X[0]), T.address_of(Y[0]), n, k)
    return main


@tilelang.jit(target=TARGET, out_idx=[2])
def silu_mul(n):
    @T.prim_func
    def main(G: T.Tensor((n,), DT), U: T.Tensor((n,), DT), O: T.Tensor((n,), DT)):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
            for i in T.Parallel(256):
                idx = bx * 256 + i
                O[idx] = (G[idx] / (1.0 + T.exp(-G[idx]))) * U[idx]
    return main


raw_g, wg = make_q4(FFN, HID)
raw_u, wu = make_q4(FFN, HID)
raw_d, wd = make_q4(HID, FFN)
pg, pu, pd = repack(raw_g, FFN, HID), repack(raw_u, FFN, HID), repack(raw_d, HID, FFN)
tg, tu, td = (torch.from_numpy(p) for p in (pg, pu, pd))
kg, kd = gemv(FFN, HID), gemv(HID, FFN)
ks = silu_mul(FFN)

x = torch.randn(HID)


def fwd():
    g = kg(tg, x)
    u = kg(tu, x)
    s = ks(g, u)
    return kd(td, s)


y = fwd()
sref = torch.nn.functional.silu(torch.from_numpy(wg) @ x) * (torch.from_numpy(wu) @ x)
ref = torch.from_numpy(wd) @ sref
err = (y - ref).abs().max().item() / ref.abs().max().item()
print(f"rel_err={err:.3e}")
for _ in range(3):
    fwd()
t0 = time.perf_counter()
it = 20
for _ in range(it):
    fwd()
dt = (time.perf_counter() - t0) / it
print(f"decode FFN (IME2): {dt*1e3:.3f} ms/token")
print("PASS" if err < 3e-2 else "FAIL")
