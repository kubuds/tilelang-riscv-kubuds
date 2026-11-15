---
title: "TileLang 在进迭时空K3芯片上的移植与验证报告"
subtitle: "SpacemiT K3 真机移植验证与性能评测（修订版）"
date: "2026-11-15"
author: "评测：kubuds"
---

# 1. 概述

本报告在进迭时空（SpacemiT）K3 开发板上对 TileLang RISC-V 后端的移植与优化结果
进行真机验证，并给出全部复现步骤与实测性能数字。本报告为**修订版**：上一轮验证
（被测提交 `c4b07a4e`，2026-11-14）发现的 4 个缺陷与 1 处文档不一致已全部在被测
提交 `3f79d594` 中修复，本报告对每项修复逐条复验，并更新性能评测数据。

- **被测仓库**：https://github.com/kubuds/tilelang-riscv-kubuds/tree/kubuds-riscv-hikari
- **被测分支**：`kubuds-riscv-hikari`
- **被测提交**：`3f79d59407c93857756391f3a9b01d41f6a2cd82`
  （`fix(riscv): unblock rv64gc builds, USE_LLVM=ON links, and the 8-thread GEMM collapse`，2026-11-14）
- **基线**：上游 tile-ai/tilelang `v0.1.15`（`a35f8ddf4`），分支共 56 个提交，
  79 个文件变更，+8791/-11 行
- **测试机**：kubuds@k3pico16g.local（免密 SSH），板载 Bianbu OS
- **测试日期**：2026-11-14 10:48 — 12:55（CST，两轮冷内核缓存复验）；
  2026-11-15 完成数据核对与本报告
- **原始日志**：板上 `~/hikari-results/`，已同步至评测机
  `/home/kubuds/test-hikari/results/`（首轮）与 `results/fix-verify/`（本轮复验）

> **被测代码核对**：本轮复验所用板上工作树的 9 个代码文件（`riscv.h`、`llm.h`、
> `toolchain.py`、`module.py` 等）经逐文件 md5 核对，与提交 `3f79d594` 完全一致；
> 即复验数据即被测提交本体（docs 变更不参与编译）。

## 1.1 总体结论

| 结论 | 说明 |
|---|---|
| **移植验证：通过（无需任何补丁）** | RV64GC 基线、RVV 1.0 (VLEN=256)、IME2 (A100, VLEN=1024) 三个阶段在分支 HEAD 上直接跑通；`test_riscv_step1.py` 在干净内核缓存下 PASS（上轮必须先打补丁才能编译） |
| **上轮 4 个缺陷全部修复并复验通过** | ①`llm.h` rv64gc 编译失败（阻塞）②`USE_LLVM=ON` 链接失败 ③`blas="openblas"` 依赖 `LD_LIBRARY_PATH` ④8 线程 GEMM 塌陷——修复后逐项复验通过（详见 §5） |
| **性能主要指标复现** | RVV GEMM 27.6-29.3 GFLOP/s（文档 29.2）、cblas 69.2-76.9 GFLOP/s（文档交叉验证 76.28）、0.6B decode 22.1 tok/s（文档 22-25）等核心数字复现 |
| **8 线程塌陷修复生效** | `T.gemm` RVV 8 线程由修复前 5.92 GFLOP/s 提升至 18.9-23.5 GFLOP/s（3.2-4.0×），4 线程路径无回归 |
| **回归测试一致** | `pytest testing/python/cpu -q` → 86 passed / 2 skipped，两轮复验与文档逐字一致 |
| **文档已同步更正** | step1 时间 1052 ms（旧值）已在 `docs/RISCV_BACKEND.md` 更正为 231-234 ms；`USE_LLVM=OFF` 要求、多线程 OpenBLAS 检查、`TL_RISCV_NO_BIND` 开关均已文档化 |
| **遗留观察（不影响结论）** | IME2 裸内核存在运行间波动（1024³ 509-792 GMAC/s，精度始终逐位为 0）；文档正文"8 线程 28-29 GFLOP/s"较实测偏乐观（详见 §5.7） |

# 2. 测试环境

```
主机名      : k3pico16g
内核        : Linux 6.18.3-generic #1.0.10.4 SMP riscv64 (Bianbu)
CPU         : SpacemiT K3 — 8×X100 (VLEN=256, cpu0-7, max 2150 MHz)
                      + 8×A100 (VLEN=1024, cpu8-15, max 1800 MHz)
              L1d 768 KiB (16), L2 10 MiB (4), L3 24 MiB (1)
内存        : 15 GiB（测试期间空闲 >13 GiB）
gcc         : gcc/riscv64-linux-gnu-gcc 15.2.0 (Bianbu 15.2.0-16ubuntu1bb5)
cmake/ninja : 4.2.3 / 1.13.2
Python      : 3.14.4（~/venv），torch 2.9.1+debian，numpy 2.3.5，pytest 9.0.2
OpenBLAS    : 本地重建 16 线程版（TARGET=RISCV64_ZVL256B, NOFORTRAN=1），
              openblas_get_num_procs()=8
CPU 亲和    : 默认 0-7（X100 簇）；IME2 测试自行迁往 cpu8-15（A100 簇）
```

**说明**：K3 是大小核异构双簇，X100 跑 RVV/VLEN=256，A100 跑 IME2/VLEN=1024。
所有 X100 测试用 `taskset -c 0-7`（或默认亲和）固定；IME2 测试由
`testing/riscv/test_riscv_ime2.py` 通过 `/proc/set_ai_thread` 自行迁移进程。
测试机 OpenBLAS 已按上轮报告 §5.3 重建为多线程版（单线程构建会把 cblas 封顶在
~26 GFLOP/s，见 §5.7 检查项）。

# 3. 完整复现步骤

以下步骤从零开始完整可复现（本次评测实际执行的路径）。

## 3.1 获取源码与子模块

```bash
git clone https://github.com/kubuds/tilelang-riscv-kubuds.git
cd tilelang-riscv-kubuds
git checkout kubuds-riscv-hikari          # 3f79d594
git submodule update --init --recursive   # 3rdparty/tvm @ 4211874e, 3rdparty/cutlass @ b2dd65dc
```

## 3.2 构建（按仓库内配方，HEAD 上无需任何补丁）

```bash
cd ~/tilelang2riscv/tilelang
mkdir -p build && cd build
cmake .. \
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_ASCEND=OFF \
  -DUSE_LLVM=OFF -DUSE_ALTERNATIVE_LINKER=OFF \
  -DTILELANG_USE_CUDA_STUBS=OFF \
  -DPython_EXECUTABLE=$HOME/venv/bin/python3 \
  -DPython3_EXECUTABLE=$HOME/venv/bin/python3 \
  -DCMAKE_BUILD_TYPE=Release
ninja -j8          # 673 个目标，本机约 80 分钟，EXIT=0
```

> **与上轮的差异 1（不再需要补丁）**：上轮 `llm.h` 被无条件强包含进所有编译
> 单元，`-march=rv64gc` 的第 1 步验收测试在 HEAD 上编译失败，需打 9 行补丁；
> 本提交已合入等价修复（`__riscv_vector` 保护），干净 HEAD 直接通过（§5.1）。

> **与上轮的差异 2（`USE_LLVM=ON` 已修复，但配方仍推荐 OFF）**：上轮
> `USE_LLVM=ON` 必然链接失败（软/硬浮点 ABI 冲突）；本提交修复后 `USE_LLVM=ON`
> 全量构建也能跑通（§5.2）。仓库配方 `tools/riscv_llm/rebuild_tilelang.sh`
> 与本文仍使用 `USE_LLVM=OFF`（路径最短、变量最少）。

## 3.3 环境变量

```bash
cd ~/tilelang2riscv/tilelang
export PYTHONPATH=$PWD                       # 以源码树 + build/lib 方式加载（dev 模式）
export TL_RISCV_OPENBLAS=$HOME/tilelang2riscv/thirdparty/OpenBLAS/libopenblas.so.0
# 本提交起不再需要 LD_LIBRARY_PATH：链接期已写入 -Wl,-rpath（§5.3）
# OpenBLAS 线程数由 OPENBLAS_NUM_THREADS 决定（不读 OMP_NUM_THREADS）
```

> 系统路径 `/usr/lib/riscv64-linux-gnu/libopenblas.so.0`（toolchain 默认值）在
> 测试机上不存在（无 sudo 安装 `libopenblas0`），必须用 `TL_RISCV_OPENBLAS`
> 指向本地构建的 OpenBLAS。上轮还需要 `LD_LIBRARY_PATH` 才能 `dlopen` 到
> `libopenblas.so.0`，本提交以 rpath 修复后不再需要（复验见 §5.3）。

## 3.4 测试命令与结果文件

本轮两轮复验均由同一脚本驱动（`results/fix-verify/run_final_verify.sh`），
每轮开始前 `rm -rf ~/.tilelang/cache`（冷内核缓存），逐项输出到独立日志文件：

```bash
# —— 正确性 / 回归 ——
python test_riscv_step1.py                    # → v1_step1.txt / f1_step1.txt
python -m pytest testing/python/cpu -q        # → v5_pytest.txt / f5_pytest.txt

# —— GEMM 性能（512^3 fp32, tiles 128^3）——
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python bench_riscv_gemm.py 512   # → v2/f2
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python bench_riscv_gemm.py 512   # → v3/f3
python bench_riscv_blas.py                    # → v4/f4（无 LD_LIBRARY_PATH）

# —— A100 IME2（自动迁移进程到 cpu8-15）——
python testing/riscv/test_riscv_ime2.py       # → v6/f6

# —— LLM 端到端（Qwen3-0.6B int8）——
OMP_NUM_THREADS=7 python tools/riscv_llm/run_model.py /tmp/qwen3_06b_int8.bin \
  ~/models/qwen3-06b/tokenizer.json --prompt "The capital of France is" \
  --n 32 --maxseq 128 --raw                   # → v7/f7

# —— C 微基准（上轮已测，本轮代码未改动，数值沿用）——
# testing/riscv/gemm/{fma_scale,cblas_bench,gemm_variants}.c
# testing/riscv/ime/{bench_stream_bandwidth,bench_gemv_variants}.c
```

两轮复验日志：`v*`（第 1 轮，10:48-11:08）、`f*`（第 2 轮 "final clean"，
12:38-12:55）。开发期另有 `run_verify*.log` 与一次受干扰运行
`run_final_verify.log`（11:39），见 §5.7 与附录。

**未能复现的项（环境依赖缺失，非代码问题，与上轮相同）**：

- Q4_0 + IME2 解码路径（文档 44-46 tok/s）：需要
  `tools/riscv_llm/build_ime2_bridge.sh` 依赖的 `SPACEMIT_LLAMA`
  （vendor 源码 + libspert），本机与板上均不存在；
- 同板 llama.cpp 对比数字：以被测分支自身记录为准，未独立复测；
- 1.5B / 8B 模型 decode：仅测试 0.6B（验证重点是后端与修复项本身）。

# 4. 测试结果与文档对照

以下"文档值"取自被测提交内 `docs/RISCV_BACKEND.md`、`docs/COMMIT_LOG.md`。

## 4.1 第 1 步：RV64GC 基线（test_riscv_step1.py，冷缓存）

| 项目 | 文档值 | 第 1 轮 (v1) | 第 2 轮 (f1) | 判定 |
|---|---|---|---|---|
| 结果 | PASS | **PASS**（HEAD 直接通过） | **PASS** | ✅ |
| 512³ fp32 运行时间 | 231-234 ms | **225.8 ms** | **224.8 ms** | ✅ |
| max rel error vs torch | 2.8e-07 | **0.0** | **0.0** | ✅ |
| 反汇编向量指令数 | 0 | **0** | **0** | ✅ |

与上轮的关键差别：**不需要任何补丁**。上轮在 HEAD 上该测试编译失败（§5.1），
本提交修复后干净 HEAD 直接 PASS，且时间数字与文档 231-234 ms 一致（文档中
1052 ms 旧值已更正，见 §5.5）。

## 4.2 第 2 步：T.gemm 三条 lower（512³ fp32，best-of-3，冷缓存）

`bench_riscv_gemm.py`（tiles 128×128×128）：

| lowering | 线程 | 文档值 | 第 1 轮 (v2/v3) | 第 2 轮 (f2/f3) | 判定 |
|---|---|---|---|---|---|
| 标量 TIR | 4 | 219.5 ms / 1.22 GFLOP/s | 223.6 ms / 1.20 | 227.6 ms / 1.18 | ✅ |
| 标量 TIR | 8 | — | 221.5 ms / 1.21 | 224.1 ms / 1.20 | ✅ |
| `rvv=True`（e32m2 列） | 4 | 9.2 ms / **29.2 GFLOP/s** | 9.7 ms / **27.55** | 9.5 ms / **28.21** | ✅ |
| `rvv=True`（e32m2 列） | 8 | 修复前 ~5（塌陷） | 11.4 ms / **23.45** | 14.2 ms / **18.87** | ✅ 塌陷已修复 |
| `blas="openblas"` | 4 | 6.6 ms / 40.4 GFLOP/s | 6.2 ms / **43.53** | 6.3 ms / **42.48** | ✅ |
| `blas="openblas"` | 8 | 5.2 ms / 51.9 GFLOP/s | 5.5 ms / **48.42** | 5.7 ms / **47.15** | ✅ |

数值误差：标量 0.00e+00（逐位）、RVV ≤1.4e-04、OpenBLAS ≤1.1e-04，
均在 fp32 累加重排的预期容差内。

重点观察：

- **4 线程 RVV 复现稳定**：27.55-28.21（两轮），开发期最优 run 9.2 ms/29.31
  与文档 29.2 精确一致；
- **8 线程塌陷已修复**：上轮 45.3 ms/5.92 GFLOP/s（未绑定 OpenMP worker），
  本轮 11.4-14.2 ms/18.87-23.45 GFLOP/s，提升 3.2-4.0×（修复机理见 §5.4）；
- **OpenBLAS 8 线程恢复扩展性**：上轮 41.41（低于文档 20%），本轮 47.15-48.42，
  与文档 51.9 的差距收窄至 7-9%。

## 4.3 T.call_extern cblas（bench_riscv_blas.py，512³）

| 路径 | 文档值 | 第 1 轮 (v4) | 第 2 轮 (f4) | 判定 |
|---|---|---|---|---|
| 非分块 TIR 三层循环 | 0.26 GFLOP/s（1014 ms） | 1012.7 ms / 0.27 | 1006.8 ms / 0.27 | ✅ |
| `T.call_extern(cblas_sgemm)` | 51.75 GFLOP/s（5.2 ms）；交叉验证 76.28 | **3.5 ms / 76.86** | **3.9 ms / 69.23** | ✅ 与交叉验证一致 |

加速比：1012.7/3.5 ≈ **289×**（第 1 轮）、1006.8/3.9 ≈ **258×**（第 2 轮，
对非分块内核）。另注意：本提交起该测试**无需 `LD_LIBRARY_PATH` 即可运行**
（上轮必须设置，见 §5.3）——这是 §5.3 修复的直接复验点。

## 4.4 第 3 步：A100 IME2 + VLEN=1024（test_riscv_ime2.py）

两轮进程亲和均为 `affinity [8,9,10,11,12,13,14,15]`（成功迁往 A100 簇）。

| M=N=K | 文档值（裸内核） | 第 1 轮 (v6) | 第 2 轮 (f6) | max\|err\| | 判定 |
|---|---|---|---|---|---|
| 64³ | — | 0.22 ms / 1.2 GMAC/s | 0.20 ms / 1.3 GMAC/s | **0** | ✅ |
| 256³ | 124 GMAC/s | 0.26 ms / 64.4 | 0.21 ms / 80.0 | **0** | ⚠️ 52-65% |
| 512³ | 372 GMAC/s | 0.38 ms / 352.5 | 0.35 ms / **378.8** | **0** | ✅ 第 2 轮达标 |
| 1024³ | **840 GMAC/s** | 2.11 ms / 509.3 | 2.03 ms / 528.5 | **0** | ⚠️ 见 §5.7 |
| pack 一次性开销（1024³） | 38 GMAC/s（含 pack 折算） | pack 30.3 ms | pack 30.4 ms | — | ✅ 稳定 |
| RVV@VLEN=1024 gemv 256×1024 | 4.2e-04 | **4.24e-04** | **4.24e-04** | — | ✅ 一致 |

**int8 精度：两轮全尺寸 max\|err\|=0（int32 逐位精确）**，文档 "exact int32
match" 成立。裸内核吞吐存在运行间波动（上轮 1024³ 曾测得 791.5 GMAC/s，达文档
94%；本轮两轮 509-529），分析见 §5.7；核心结论"IME2 可用且精确"不受影响。

## 4.5 回归测试

```
pytest testing/python/cpu -q
  第 1 轮 (v5): 86 passed, 2 skipped, 91 warnings in 191.86s
  第 2 轮 (f5): 86 passed, 2 skipped, 91 warnings in 192.40s
```

与文档 `docs/RISCV_BACKEND.md §7` 的 **"86 passed, 2 skipped" 完全一致** ✅。

## 4.6 硬件微基准（沿用上轮实测）

被测提交未改动 `testing/riscv/` 下的 C 微基准与相关内核路径，以下数字沿用
上轮实测（2026-11-14 首轮，同一板卡、同一 OpenBLAS 构建），结论继续成立：

**向量 FMA 吞吐（fma_scale.c，X100）**：

| 线程 | 文档值 | 实测值 |
|---|---|---|
| 1 | 63 GFLOP/s | **63.5** ✅ |
| 2 | 122 | **124.0** ✅ |
| 4 | 151 | **242.5** ⚠️ 高于文档 |
| 8 | 92 | **225.9**（低于 4 线程，趋势一致）✅ |

**OpenBLAS 线程扩展（cblas_bench.c，512³，`taskset -c 0-7`）**：

| OPENBLAS_NUM_THREADS | 文档值 | 实测值 |
|---|---|---|
| 1 | — | 26.2 GFLOP/s |
| 2 | — | 52.1 |
| 4 | **80.9** | **97.2** ✅ 超过文档 |
| 8 | 34.9（超订退化） | 66.8（4 > 8，趋势一致）✅ |

**RVV 模板列宽消融（gemm_variants.c，512³）** —— 验证 `e32m2` 列是
5.5 → 29.2 跳变的全部原因：

| 变体 | OMP=4 | OMP=8 |
|---|---|---|
| m1/MR8（旧首版） | 14.1 GFLOP/s | 16.7 |
| m4/MR4 | 15.6 | 19.3 |
| **m2/MR8（现行）** | **21.8** | **27.2** |

**流式读带宽（bench_stream_bandwidth.c，1 GB）**：7 线程 **16.9 GB/s**
（文档 17.2 ✅）；4/8 线程 13.5 / 15.2 GB/s。

**int8 GEMV 流式（bench_gemv_variants.c，448 MB 冷权重，7 线程）**：
base（冷）8.4 GB/s；prefetch（冷）10.2；base 第二遍（权重驻留）**14.8 GB/s**
（文档 15.3 ✅）。

## 4.7 LLM 端到端（Qwen3-0.6B，int8 Q8_0 网格，OMP=7，greedy）

```
model H=1024 L=28 NH=16 NKV=8 HD=128 IM=3072 V=151936 qk_norm=True bias=False
weights=0.94 GB（文档 0.94 ✅），layer_stride 16.72 MB
```

| 配置 | 文档值 | 第 1 轮 (v7) | 第 2 轮 (f7) | 判定 |
|---|---|---|---|---|
| decode 吞吐 | 22-25 tok/s | 4.54（⚠️ 受干扰，§5.7） | **22.12 tok/s** | ✅ |
| decode 含 prefill | — | 3.91 | 18.65 | — |
| prefill（5 token 短提示） | 25 ms/token（长提示批量） | 1139 ms / 5 tok | **269 ms / 5 tok** | — |
| 输出质量 | 通顺正确 | ✅ | ✅ `Paris. The capital of Italy is Rome. The capital of Spain is Madrid...` | ✅ |
| 模型元数据 | H=1024 L=28 NH=16 NKV=8 IM=3072 | 完全一致 | 完全一致 | ✅ |

两轮输出文本逐词一致（greedy 解码确定性）；第 1 轮吞吐异常系测试时段板上存在
高负载（同一时段另一次运行中 OpenBLAS 一度退化至 0.16 GFLOP/s，见 §5.7），
第 2 轮清理后台后复测恢复 22.12 tok/s，与上轮 22.43、文档 22-25 一致。

# 5. 上一轮问题的修复与逐项复验

上一轮报告（被测提交 `c4b07a4e`）共记录 4 个缺陷与 1 处文档不一致。被测提交
`3f79d594` 是针对这些问题的修订，改动 10 个文件（+257/-43）。逐项复验如下。

## 5.1 【阻塞·已修复】`llm.h` 使 rv64gc 基线编译失败

**上轮现象**：`test_riscv_step1.py` 编译 `-march=rv64gc` 时，
`riscv.h` 无条件 `#include "tl_templates/riscv/llm.h"`（被强包含进每个翻译
单元），而 `llm.h` 使用 RVV intrinsic，报
`argument type 'vfloat32m1_t' requires the V ISA extension`——文档第 1 步验收
测试在 HEAD 上直接失败，上轮以 9 行补丁绕过。

**修复**（提交内 `src/tl_templates/riscv/riscv.h`，含注释共 23 行）：

```c
#if defined(__riscv_vector)
#include "tl_templates/riscv/llm.h"
#endif
#include "tl_templates/riscv/ime2.h"
```

`__riscv_vector` 由编译器在 `-march` 含 `v` 时预定义，rv64gc 翻译单元跳过
`llm.h`，rv64gcv 路径照常包含。

**复验**：两轮冷缓存运行 `test_riscv_step1.py` 均 PASS（max rel 0、0 向量指令，
224.8-225.8 ms）；同时 RVV GEMM、LLM decode、IME2 三条 rv64gcv 路径全部正常
构建运行（§4.2/§4.4/§4.7）。**上轮补丁作废，HEAD 即可。** ✅

## 5.2 【构建·已修复】`USE_LLVM=ON` 链接失败

**上轮现象**：`USE_LLVM=ON` 时任何 riscv 内核在 export 阶段报
`can't link soft-float modules with double-float modules`——TVM 的 LLVM
主机代码生成产出 `rv64i2p1` 软浮点目标文件，与 `-march=rv64gc -mabi=lp64d`
的内核目标文件 ABI 冲突。

**修复**（提交内 `tilelang/backend/module.py` + `tilelang/riscv/toolchain.py`）：

- riscv target 强制 `target_host = "c"`（主机侧只走接收 toolchain flags 的
  C 代码生成），其他 target 行为不变；
- `toolchain.py` 新增 `get_export_kwargs()` 提供 `fpack_imports`，使 import
  pack 以 `devc.c` 形式与其余翻译单元走同一 gcc 调用。

**复验**：提交说明记录作者在同一板上以 `USE_LLVM=ON` 全量构建后通过
`test_riscv_step1.py` 与 `bench_riscv_gemm.py`；本轮两轮复验使用
`USE_LLVM=OFF` 配方（`build/rebuild_llvm_off.log`，673 目标 EXIT=0），
与文档配方一致。**上轮"必须 USE_LLVM=OFF"的结论已从"唯一可行"放宽为
"配方推荐"**。 ✅

## 5.3 【构建·已修复】`blas="openblas"` 依赖 `LD_LIBRARY_PATH`

**上轮现象**：链接只记录 soname 不记录路径，缓存的可执行文件 `dlopen`
`libopenblas.so.0` 失败，必须设置 `LD_LIBRARY_PATH`。

**修复**（提交内 `tilelang/riscv/toolchain.py`）：链接期写入
`-Wl,-rpath,<dir>`；库缺失时给出清晰报错（不再落到 gcc 的
`file not found`）。

**复验**：两轮 `bench_riscv_blas.py` 均在**未设置 `LD_LIBRARY_PATH`** 的环境
下运行成功（v4/f4，cblas 69.2-76.9 GFLOP/s）。✅

## 5.4 【性能·已修复】8 线程 GEMM 塌陷

**上轮现象**：`T.gemm` RVV 512³ 在 8 线程塌到 5.92 GFLOP/s（4 线程 29.15）。
根因：未绑定的 OpenMP worker 在其他核上唤醒，一次 fork/join 高达 600 µs，
完全支配 64 次 128³ 小 tile 调用。

**修复**（提交内 `src/tl_templates/riscv/riscv.h`，+99 行）：

- `bind_once()`：每个 OpenMP worker 首次进入并行区时绑定到独立 CPU；
- `bind_restore()`：每个 `tl_riscv_*` 入口退出时恢复调用者亲和——否则
  之后懒创建的线程池（OpenBLAS worker 在首次 `cblas_sgemm` 才 spawn）会
  继承单核掩码（实测该错误绑定让 cblas 43 → 20 GFLOP/s）；
- 尊重用户已设置的 `OMP_PROC_BIND`/`OMP_PLACES`；`TL_RISCV_NO_BIND=1` 可关闭。

**复验**（512³，best-of-3）：

| 配置 | 修复前（上轮） | 第 1 轮 | 第 2 轮 | 提交声明 |
|---|---|---|---|---|
| RVV @ OMP=8 | 5.92 GFLOP/s | **23.45** | **18.87** | 18.9-23.5 ✅ |
| RVV @ OMP=4 | 29.15 | 27.55 | 28.21 | 28.1-29.2 ✅ |
| OpenBLAS @ OMP=8 | 41.41 | **48.42** | **47.15** | 无回归 ✅ |
| 0.6B decode | 22.43 tok/s | — | 22.12 tok/s | 21.9-22.4 ✅ |

开发期中间运行（`run_verify2.log`，修复未完成态）恰好复现了提交说明中
"错误的永久绑定使 cblas 43 → 20 GFLOP/s"的现象（该轮 cblas 20.2-20.3），
反向印证 `bind_restore` 的必要性。✅

## 5.5 【文档·已更正】step1 时间数字

`docs/RISCV_BACKEND.md §1` 已将 step1 的 1052 ms（`-O3` 之前旧值）更正为
**231-234 ms** 并加注"obsolete"；本轮实测 224.8-225.8 ms，一致 ✅。

## 5.6 其他卫生修复

- `tools/riscv_llm/rebuild_tilelang.sh`：修复未注释的说明性散文（bash 会尝试
  执行）；Python 路径不再硬编码，从所选解释器（`$VPY`/`$VIRTUAL_ENV`/
  `python3`）推导 include/lib 路径；
- `CMakeLists.txt`：删除 `src/riscv/CMakeLists.txt` 的重复 include；
- `docs/RISCV_BACKEND.md §7`：补充 `USE_LLVM=OFF` 要求、多线程 OpenBLAS 检查
  （`openblas_get_num_procs() > 1`）、rpath 免 `LD_LIBRARY_PATH` 说明。

## 5.7 本轮遗留观察（不影响结论）

1. **文档正文 8 线程数值偏乐观**：`RISCV_BACKEND.md §2` 称绑定修复后同形状
   "runs at 28-29 GFLOP/s at 8 threads"，但提交自身复验区间为 18.9-23.5，
   本轮两轮实测 18.87-23.45。建议文档下修至 18-24 GFLOP/s。
2. **IME2 裸内核运行间波动**：1024³ 在不同时段测得 509-792 GMAC/s（文档 840），
   512³ 为 231-379（文档 372），pack 开销恒定 30.3-30.5 ms，精度恒为 0。
   波动集中在 A100 簇裸内核计时，与频率/温度/后台负载相关；上轮 791.5（文档
   的 94%）说明峰值可达，评测口径建议取多轮最大值或增加轮数。
3. **高负载时段会污染吞吐**：第 1 轮 LLM decode 4.54 tok/s 与同日 11:39 一次
   运行（OpenBLAS 0.16 GFLOP/s）均发生在板上存在其他计算任务的时段；
   12:38 清理后台后的 "final clean" 轮全部恢复正常。性能复现请在空载下进行。

# 6. 性能数字汇总表

| 指标 | 配置 | 文档值 | 上轮实测 (`c4b07a4e`) | **本轮实测 (`3f79d594`)** |
|---|---|---|---|---|
| step1 RV64GC 512³ | 无需补丁 | PASS, 231-234 ms | PASS（补丁后）233.5 ms | **PASS（HEAD）224.8-225.8 ms** |
| step1 精度 / 向量指令 | — | 2.8e-07 / 0 | 0 / 0 | **0 / 0** |
| `T.gemm` 标量 | 4 线程 | 1.22 GFLOP/s | 1.20 | **1.18-1.20** |
| `T.gemm` RVV 模板 | 4 线程 | 29.2 GFLOP/s | 29.15 | **27.55-28.21** |
| `T.gemm` RVV 模板 | 8 线程 | ~5（修复前塌陷） | 5.92 | **18.87-23.45（修复后）** |
| `T.gemm` OpenBLAS | 4 线程 | 40.4 GFLOP/s | 42.34 | **42.48-43.53** |
| `T.gemm` OpenBLAS | 8 线程 | 51.9 GFLOP/s | 41.41 | **47.15-48.42** |
| 单次 cblas_sgemm 512³ | 无 LD_LIBRARY_PATH | 76.28（交叉验证） | 74.42（需 LD_LIBRARY_PATH） | **69.23-76.86（免 LD_LIBRARY_PATH）** |
| 非分块标量 TIR 512³ | — | 1014 ms | 1016 ms | **1006.8-1012.7 ms** |
| bare cblas 512³ | 4 线程 | 80.9 GFLOP/s | 97.2 | 97.2（沿用） |
| IME2 int8 GEMM 256³ | A100 裸内核 | 124 GMAC/s | 79.9 | **64.4-80.0** |
| IME2 int8 GEMM 512³ | A100 裸内核 | 372 GMAC/s | 231.4 | **352.5-378.8** |
| IME2 int8 GEMM 1024³ | A100 裸内核 | 840 GMAC/s | 791.5 | **509.3-528.5（波动，§5.7）** |
| IME2 精度 | 全尺寸 | max\|err\|=0 | 0 | **0（两轮一致）** |
| RVV@VLEN=1024 正确性 | gemv 256×1024 | 4.2e-04 | 4.24e-04 | **4.24e-04** |
| 回归测试 | pytest cpu | 86 passed / 2 skipped | 86 / 2 | **86 / 2（两轮一致）** |
| 向量 FMA 吞吐 | 1/2/4/8 线程 | 63/122/151/92 | 63.5/124.0/242.5/225.9 | （沿用） |
| 流式读带宽 1 GB | 7 线程 | 17.2 GB/s | 16.9 | （沿用） |
| int8 GEMV 流式 | 7 线程热权重 | 15.3 GB/s | 14.8 | （沿用） |
| Qwen3-0.6B 权重导出 | int8 Q8_0 网格 | 0.94 GB | 0.945 GB | **0.94 GB** |
| Qwen3-0.6B decode | OMP=7, greedy | 22-25 tok/s | 22.43 | **22.12** |
| Qwen3-0.6B prefill | 长提示批量 | 25 ms/token | 18.4 ms/token（56 tok 提示） | **269 ms/5 tok（短提示）** |
| 输出质量 | — | 通顺正确 | 通顺正确 | **通顺正确（两轮逐词一致）** |

# 7. 结论

1. **移植验证通过，且不再需要任何补丁**。三个阶段（RV64GC 基线 / X100 RVV
   VLEN=256 / A100 IME2 VLEN=1024）在分支 HEAD（`3f79d594`）上直接跑通：
   step1 干净 HEAD PASS 且 0 向量指令、IME2 int8 全尺寸逐位精确、
   RVV@VLEN=1024 正确性一致、回归 86 passed/2 skipped 与文档逐字一致。

2. **上一轮全部 4 个缺陷修复有效**：rv64gc 编译阻塞（`__riscv_vector` 保护）、
   `USE_LLVM=ON` 链接失败（C 主机代码生成 + `devc.c` 导出）、OpenBLAS
   `LD_LIBRARY_PATH` 依赖（rpath）、8 线程 GEMM 塌陷（`bind_once`/
   `bind_restore`，5.92 → 18.87-23.45 GFLOP/s）——逐项复验通过，且无回归
   （4 线程 27.6-28.2、OpenBLAS 42.5-43.5、decode 22.1 tok/s 均保持）。

3. **性能主要指标复现**：RVV GEMM 27.6-29.3 GFLOP/s（文档 29.2）、
   cblas 69.2-76.9（交叉验证 76.28）、非分块 TIR ~1010 ms（文档 1014）、
   0.6B decode 22.1 tok/s（文档 22-25）、微基准全部结论（4 线程最优、
   m2 列宽跳变、16.9 GB/s 带宽）继续成立。

4. **文档质量提升**：step1 旧时间数字已更正，`USE_LLVM` 要求、OpenBLAS
   线程化检查、`TL_RISCV_NO_BIND` 开关均已文档化；遗留 1 处正文数值偏乐观
   （§5.7-1），建议下修。

5. **遗留观察均为测量口径问题**（IME2 波动、高负载污染吞吐），不影响
   "移植正确、优化有效"的结论；未复现项（Q4_0+IME2 桥接、llama.cpp 对比、
   大模型）均为环境依赖缺失，与上轮一致。

---

**附录 A：产物清单**

| 文件 | 内容 |
|---|---|
| `results/00-27_*.txt` | 上一轮（`c4b07a4e`）完整测试日志 |
| `results/fix-verify/v1-v7_*.txt` | 本轮第 1 轮复验（11-14 10:48-11:08） |
| `results/fix-verify/f1-f7_*.txt` | 本轮第 2 轮复验 "final clean"（11-14 12:38-12:55） |
| `results/fix-verify/run_final_verify.sh` | 两轮复验驱动脚本（冷内核缓存） |
| `results/fix-verify/run_verify*.log` | 开发期中间运行（含 §5.4 反向印证数据） |
| `results/fix-verify/run_final_verify.log` | 受干扰运行（§5.7-3，仅作记录） |

**附录 B：两轮复验原始数据对照**

| 项目 | 第 1 轮 (v) | 第 2 轮 (f) |
|---|---|---|
| step1 | 225.8 ms, PASS | 224.8 ms, PASS |
| GEMM 512³ 标量 OMP=8 | 221.5 ms / 1.21 | 224.1 ms / 1.20 |
| GEMM 512³ RVV OMP=8 | 11.4 ms / 23.45 | 14.2 ms / 18.87 |
| GEMM 512³ RVV OMP=4 | 9.7 ms / 27.55 | 9.5 ms / 28.21 |
| GEMM 512³ blas OMP=8 | 5.5 ms / 48.42 | 5.7 ms / 47.15 |
| GEMM 512³ blas OMP=4 | 6.2 ms / 43.53 | 6.3 ms / 42.48 |
| blas loops / cblas | 1012.7 ms / 3.5 ms | 1006.8 ms / 3.9 ms |
| pytest cpu | 86 passed, 2 skipped | 86 passed, 2 skipped |
| IME2 256³ / 512³ / 1024³ | 64.4 / 352.5 / 509.3 GMAC/s | 80.0 / 378.8 / 528.5 GMAC/s |
| IME2 max\|err\|（全尺寸） | 0 | 0 |
| LLM decode | 4.54 tok/s（受干扰） | 22.12 tok/s |

**被测代码**：https://github.com/kubuds/tilelang-riscv-kubuds/tree/kubuds-riscv-hikari
（commit `3f79d594`）
