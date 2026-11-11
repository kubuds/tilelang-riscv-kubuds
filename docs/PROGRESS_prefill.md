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

## 2026-11-11 18:40 已接进引擎

`build_layer_m4` 上线，`layout=ime2q4` 自动 MR=4，两个入口（decode 的 `ime2_gemv_q4`
和 prefill 的 `ime2_gemm_q4_m`）合进同一个 `libime2bridge.so`，不再需要 LD_PRELOAD。

0.6B Q4_0、27 token 提示词、OMP=7 实测：

| 路径 | prefill | 说明 |
|---|---:|---|
| RVV 批量（M=8） | 25 ms/token | 旧 |
| Q4 不批量 | 74 ms/token | 更早 |
| **IME2 M=4** | **13.4 ms/token** | 362 ms / 27 token，其中 3 token 尾巴没凑满一个 chunk |

按 GEMM 外推应该是 8.6，差的那 5 ms 在逐行算子：rope 与 qk_norm 是**按 head 发调用**
（每行每层 24 次），一个 4 行 chunk 每层要发 ~7.8k 次 extern call。

## 2026-11-11 19:00 测量噪声与一次被否掉的改动

同一提示词跑 5 次（84 token）：951 / 1355 / 1471 ms，**±25%**；27 token 跑 5 次
324-415 ms。板上有其他负载（load 3.1、3 个登录用户），单次对比不可信。

因此“把不足一个 chunk 的尾巴补成 4 行”这条改动（预期省一次单步 ~21 ms）在噪声里
**无法证明有效**，已回退，不留未验证的复杂度。教训：以后 A/B 一律 min-of-5，
并且先看 load。

## 2026-11-11 19:40 chunk 大小与成本拆解

原位测（`testing/riscv/ime/bench_chunk.py`，min of 7×5，无提示词噪声）：

| chunk | ms/chunk | ms/token |
|---|---:|---:|
| MR=4 | 39.1 | 9.77 |
| MR=8 | 62.2 | 7.78 |
| MR=12 | 86.0 | 7.17 |
| MR=16 | 113.9 | 7.12 |

按提示词余数选 chunk（4/8/16 三个 plan），端到端：27 token 362→326 ms，
84 token 1009→732 ms（8.7 ms/token）。

**成本拆解**（`bench_chunk_pos.py`：同一 chunk 在不同 pos 计时，斜率即 attention）：

| pos | chunk (MR=16) | ms/token |
|---|---:|---:|
| 0 | 110.4 | 6.90 |
| 64 | 124.4 | 7.77 |
| 128 | 136.9 | 8.55 |
| 256 | 164.6 | 10.29 |

attention 斜率 0.212 ms/(position·chunk)：p=128 时占 20%，p=256 时占 33%。
**pos=0 的 6.9 ms/token 里约 91% 是 GEMM**（按 microbench 的 7 个矩阵 ×28 层反推 6.3）。

## 已经排除的可能

- 尾块补满 chunk：噪声里无收益，已回退。
- perf 采样：JIT 模块无符号，且抓到别的进程，无法归因。
- M≥8 的 vendor 内核：IME1 与 IME2 都只到 4 行（`count_m>=4` 走 m4，返回 4），
  所以每 4 行就要重读一遍权重，这是 15 GB/s 有效带宽的来源。

## 下一步

1. **自写 M=16 的 smt.vmadot 内核**（唯一能突破"每 4 行重读权重"的办法）：
   目标把 6.3 ms/token 的 GEMM 压到 2 ms 量级。这是新项目，不是调参。
2. attention 批量：p=128 时占 20%，16 行一起做 QK^T 可省一半以上。
3. rope/qk_norm 整段化：收益在微秒级，优先级最低。
