"""Qwen3-0.6B attention projections on IME2 Q4_0 (decode, m=1)."""
import ctypes, time
import numpy as np
import torch
import tilelang
import tilelang.language as T

ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so", mode=ctypes.RTLD_GLOBAL)
lib = ctypes.CDLL("/work/mahiru/src/tl/libime2bridge.so")
lib.ime2_repack_q4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

HID, NH, NKV, HD = 1024, 16, 8, 128
DT = "float32"
TARGET = {"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu",
          "mcpu": "generic-rv64", "mattr": ["+m", "+a", "+f", "+d", "+v"]}
rng = np.random.default_rng(0)


def q4(N, K):
    nb = K // 32
    d = rng.uniform(0.001, 0.02, size=(N, nb)).astype(np.float16)
    qs = rng.integers(0, 256, size=(N, nb, 16), dtype=np.uint8)
    raw = np.empty((N, nb, 18), dtype=np.uint8)
    raw[:, :, :2] = d.view(np.uint8).reshape(N, nb, 2)
    raw[:, :, 2:] = qs
    pb = (N // 32) * nb * 576
    packed = np.zeros(pb, dtype=np.uint8)
    r = lib.ime2_repack_q4(ctypes.c_char_p(raw.tobytes()), packed.ctypes.data, N, K)
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


x = torch.randn(HID)
projs = {
    "q(2048x1024)": (NH * HD, HID),
    "k(1024x1024)": (NKV * HD, HID),
    "v(1024x1024)": (NKV * HD, HID),
    "o(1024x2048)": (HID, NH * HD),
}
total = 0.0
for name, (n, k) in projs.items():
    p = torch.from_numpy(q4(n, k))
    kern = gemv(n, k)
    xx = torch.randn(k)
    kern(p, xx)
    for _ in range(3):
        kern(p, xx)
    t0 = time.perf_counter()
    it = 30
    for _ in range(it):
        kern(p, xx)
    dt = (time.perf_counter() - t0) / it
    total += dt
    print(f"{name}: {dt*1e3:.3f} ms")
print(f"attention projections total: {total*1e3:.3f} ms/token")
print(f"layer budget: attn proj {total*1e3:.3f} + ffn 0.681 = {total*1e3+0.681:.3f} ms")
print(f"28 layers + lm_head 4.649 = {(total*1e3+0.681)*28+4.649:.2f} ms -> {1000/((total*1e3+0.681)*28+4.649):.1f} tok/s")
