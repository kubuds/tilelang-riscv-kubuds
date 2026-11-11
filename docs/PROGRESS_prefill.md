# prefill 加速：进度台账

目标：0.6B prefill 从 25 ms/token 降到 llama.cpp 同板水平（3.3 ms/token）。
路线：Q4_0 权重 + A100 IME2 的 M=4/M=8 批量 GEMM（decode 已用 M=1 到 44-46 tok/s）。

## 2026-11-11 12:05

**已通**：`ime2_gemm_q4_m`（M=4/M=8）与 4×/8× M=1 **逐位一致**（maxdiff 0.0e+00），
在 0.6B 的真实形状上快 2.5-3.15 倍：

| N × K | 8× M=1 | M=8 | 倍数 | 备注 |
|---|---:|---:|---:|---|
| 4096×1024 (q/o) | 0.956 ms | 0.314 ms | 3.05× | 每 token 权重读一次 |
| 1024×3072 (down) | 0.673 ms | 0.271 ms | 2.48× | |
| 1024×1024 (k/v) | 0.251 ms | 0.091 ms | 2.75× | |
| 151936×1024 (head) | 31.68 ms | 10.06 ms | 3.15× | |

外推 0.6B：每 8 token 的 GEMM 成本 ≈ 3.1 × 单 token，即 **prefill ≈ 8.6 ms/token**
（现在是 25 ms/token 的 RVV 批量路径；llama.cpp 3.3）。

**关键契约**（花了两次错误才定位）：`gemm_kernel_i8i4_m4` 读的 A 不是行主序，而是
每个 K 块 `[4×f32 scale][4×int16 asum][4×32 int8]` = 152 字节，由
`spacemit_kernels::rvv::quantize_a_4row_i8` 生成；不要用 `ime1::quantize_a_4row_i8`
（那是 144 字节的另一种交错），也不要自己按行拼。M=8 = 两个 4 行组，各自一份
152 字节/块的缓冲，第二次调用 `y + 4*N`。

**另一条教训**：对比测试本身错过两次——先拿 4 行结果去比"最后一个 M=1 行的输出"，
得到 maxdiff 2.4 的假故障；kernel 其实一直是对的。

## 下一步

1. 把 `ime2_gemm_q4_m` 接进 `engine.py`：新 `build_layer_m4`（4 行 TIR 程序），
   GEMV 走 M=4，逐行算子（rmsnorm/rope/qk_norm/attention）保持逐行；head 用 M=8。
2. 用 0.6B 实测 prefill，对照 8.6 ms/token 的外推值。
3. 若仍慢，量 attention/逐行算子的占比（decode 时它占约 3 ms/token，prefill 里会被 4 行摊薄）。
