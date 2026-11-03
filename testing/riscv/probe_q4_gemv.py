"""Probe: 4-bit GEMV kernel against a numpy dequant reference."""

import numpy as np
import tilelang
import tilelang.language as T
from tilelang import tvm

N, K = 64, 256


@T.prim_func
def gemv(W: T.Tensor((N * K // 2,), "uint8"),
         S: T.Tensor((N * K // 32 * 2,), "uint8"),
         X: T.Tensor((K,), "float32"), Y: T.Tensor((N,), "float32")):
    with T.Kernel(1):
        T.call_extern("int32", "tl_riscv_gemv_q4", N, K,
                      T.address_of(W[0]), T.address_of(S[0]),
                      T.address_of(X[0]), T.address_of(Y[0]))


def main():
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)
    blocks = w.reshape(N, K // 32, 32)
    sc = np.abs(blocks).max(axis=2) / 7.0
    q = np.clip(np.round(blocks / sc[:, :, None]), -8, 7).astype(np.int8) + 8
    q = q.astype(np.uint8).reshape(N, K // 32, 2, 16)
    packed = (q[:, :, 0, :] | (q[:, :, 1, :] << 4)).astype(np.uint8).reshape(N, K // 2)
    x = rng.standard_normal(K).astype(np.float32)

    cfg = {"kind": "riscv", "march": "rv64gcv", "mabi": "lp64d", "vlen": 256}
    with tvm.target.Target(cfg):
        f = tilelang.compile(gemv, out_idx=None, execution_backend="tvm_ffi",
                             compile_flags=["-O3"])
    y = np.zeros(N, np.float32)
    f(packed.reshape(-1), sc.astype(np.float16).view(np.uint8).reshape(-1), x, y)
    wd = (q[:, :, 0, :].astype(np.float32) - 8) * sc[:, :, None]
    wd2 = (q[:, :, 1, :].astype(np.float32) - 8) * sc[:, :, None]
    full = np.concatenate([wd, wd2], axis=2).reshape(N, K)
    ref = full @ x
    err = np.abs(y - ref).max()
    print("max abs err", err, "ref scale", np.abs(ref).max())
    assert err < 1e-3 * max(1.0, np.abs(ref).max())


if __name__ == "__main__":
    main()
