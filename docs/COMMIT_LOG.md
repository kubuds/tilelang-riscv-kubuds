# TileLang RISC-V 后端：54 个提交逐条说明（原理 + 实测）

仓库：`kubuds/tilelang-riscv-kubuds`　分支：`kubuds-riscv-hikari`（基于 tag v0.1.15）
署名：Hikari KUBUDS <hikari@kubuds.cn>（54 个提交的作者/提交者统一）
测试板：SpacemiT K3（Bianbu，8×X100 RVV VLEN=256 + 8×A100 IME2 VLEN=1024，31 GB）

> 说明：提交哈希是分支上重写署名后的值；实测数据都在 K3 上取得。
> 许可证：改动全部 Apache-2.0；OpenBLAS 链接（BSD-3-Clause）；SpacemiT vmadot 内核
> 引用 MIT；Q4_0 量化规则引自 ggml（MIT，仅算法，非拷贝）。

---

## 阶段一：第一步（RV64GC 基线）+ 第二步前半（RVV 骨架 + OpenBLAS）

### cdef5e13 feat(riscv): add RISC-V backend skeleton (v0.1.15, step 1)
- 说明：新建 `tilelang/riscv/`（backend/codegen/execution_backend/toolchain）与 `src/riscv/`，
  注册 `target={"kind":"riscv",...}`。
- 原理：复用 TVM 的 CPU pass pipeline 与 C 主机代码生成，只加 ISA 相关部分；RV64GC 是
  TVM 原生支持，第一步只是"让 C 目标在 riscv 三元组下编出来"。
- 实测：`baseline_add.py` max_err 0.0，`objdump | grep -c vsetvli` = 0（纯标量，符合预期）。

### a3473406 fix(riscv): make the RV64GC path actually build and run
- 说明：修 C 目标在 riscv64 上的编译/运行链路。
- 原理：CPU C 目标默认生成的主机代码引用了只有 CUDA/ROCm 才有的符号；riscv 路径要关掉这些。
- 实测：`baseline_gemm_global.py`（手写 TIR 三层循环，无 T.gemm）跑通。

### 06ecccba feat(riscv): step 1 baseline passes -- RV64GC, zero vector instructions
- 说明：第一步验收——RV64GC 标量基线通过。
- 原理：`-march=rv64gc` 下不产生任何向量指令，证明基线就是"TVM 的 CPU 路径在 RV64GC 上"。
- 实测：512³ fp32 手写 TIR 1052 ms，max rel err 2.8e-07（vs torch）；`vsetvli` 计数 0。

### b1af432a fix(riscv): pin VLEN the way GCC actually accepts it
- 说明：把 `-march=rv64gcv_zvl256b` 的 VLEN 写法改成 GCC 真正认的形式。
- 原理：GCC 的 `zvlNb` 后缀约束 N 必须是 2 的幂且拼写严格；文档写法与实际接受不一致。
- 实测：`-march=rv64gcv_zvl256b` 编过，RVV 目标可用。

### d09beaea perf(riscv): -O3 for the host compile (4.6x), but still not real RVV
- 说明：给主机编译加 `-O3`。
- 原理：`toolchain.py:get_compile_flags()` 默认返回空列表，生成码是 `-O0`；加 `-O3` 立竿见影。
- 实测：标量 GEMM 0.26 → 1.20 GFLOP/s（4.6×），但仍无 RVV（见下一条的 cblas 对比）。

### 7176104f feat(riscv): wire OpenBLAS into the host compile (blocked by C++ enums)
- 说明：把 cblas_sgemm 接进 `T.gemm` 的 host 编译。
- 原理：`T.call_extern("cblas_sgemm", ...)` + `-lopenblas`；卡点是 C++ 枚举类型在 extern
  原型里的冲突（见下一条，是假警报）。
- 实测：此条本身被阻塞，未出数。

### 45c68c2d docs(riscv): record step-2 blockers: extern prototype conflict, not enums
- 说明：定位上一条的"枚举冲突"其实是 extern 原型重复声明，不是 C++ 枚举。
- 原理：TileLang 生成的原型与 OpenBLAS 头文件原型重复、签名不一致 → 链接错。
- 实测：诊断性提交，无性能数据。

### c5a5d09c feat(riscv): OpenBLAS reachable -- 51.75 GFLOP/s vs 0.26 scalar (195x)
- 说明：OpenBLAS 终于接通。
- 原理：cblas_sgemm 是 GEMM 的唯一快路；标量三层循环 0.26 GFLOP/s 对比 OpenBLAS 51.75。
- 实测：512³ 标量 0.26 → cblas 51.75 GFLOP/s。

### 4c869c54 docs(riscv): correct the cblas speedup -- 44x, not 195x
- 说明：修正上一条的倍率口径。
- 原理：195× 是把"未优化标量"当分母；公平基线（-O3 标量 1.20）是 44×。
- 实测：cblas 51.75 / 标量(-O3) 1.20 ≈ 43×。

### e006ddf9 docs(riscv): T.gemm has no CPU implementation in v0.1.15
- 说明：记录关键事实——v0.1.15 的 CPU 家族没注册任何 GEMM。
- 原理：`SelectInst` 返回 `cpu.scalar` 但无人注册它，`T.gemm` 在 CPU/riscv 上直接报错。
- 实测：`T.gemm` 报 "No GEMM implementation registered"（这就是整个后端要补的第一件事）。

### 5fdd2dfe feat(riscv): first GEMM impl for the CPU family (advances the failure)
- 说明：写第一个 GEMM 实现（`infer_layout` + 标量 lower）。
- 原理：`T.gemm` 需要 fragment layout 推理与 lower；先让失败从"无实现"推进到具体报错。
- 实测：失败信息前进到 layout 推理阶段。

### eb0e015a fix(cpu): let fill accept local.fragment
- 说明：修 `fill` 不接受 `local.fragment` 的问题。
- 原理：GEMM 的 fragment 布局需要 `fill` 原语支持，是 GEMM 走通的先决小修。
- 实测：诊断性，无数据。

### a56aa628 diag(launch): name the offending buffer in the single-smem check
- 说明：把"单共享内存"检查的报错加上 buffer 名。
- 原理：CPU 类目标不支持多 shared buffer 时报错含糊，加名字才好定位。
- 实测：诊断性，无数据。

### 7108927f fix(launch): allow multiple shared buffers on CPU-class targets
- 说明：允许 CPU 类目标分配多个 shared buffer。
- 原理：GEMM 需要两个 fragment buffer，CPU 路径的单 smem 检查误伤。
- 实测：诊断性，无数据。

### a6c7b608 fix(cpp-templates): declare float2/float4 vector element types
- 说明：修 `tl_templates/cpp/common.h` 里 float2/float4 声明在 namespace 外。
- 原理：`vec_type` 在 `namespace tl` 内、float2/float4 在 namespace 外 → 向量化拷贝编不过。
- 实测：修后向量化拷贝正常编译（这是后面 `-include` 那条的真正根因之一）。

### 9274d6fd feat(riscv): T.gemm works end to end on RISC-V
- 说明：`T.gemm` 在 riscv 上端到端跑通。
- 原理：-include 强制每个 TU 包含模板头（此前只有 host TU 包含、device TU 不含 → float4 未声明）。
- 实测：128³ fp32，rv64gc 标量 max abs 0.0（bit-exact），blas 分支 1.9e-05（cblas 重排累加的签名）。

### 5d47f900 bench(riscv): T.gemm on K3 X100 -- 1.20 vs 52.27 GFLOP/s (43.6x)
- 说明：K3 上 GEMM 三路对比基准。
- 原理：标量 / RVV 模板 / OpenBLAS 三条 lower 同场。
- 实测：512³ 标量 1.20、cblas 52.27 GFLOP/s。

### 294b27f2 feat(riscv): RVV 1.0 fp32 GEMM template behind target attr rvv
- 说明：第一个 RVV GEMM 模板（`target.attrs["rvv"]` 开关）。
- 原理：N 按 256 列分块、MR=8 行共享一个 k 步；8 条累加链避开 vfmacc 延迟墙。
- 实测：512³ 模板 48.5 ms / 5.53 GFLOP/s（首版，后续 c9e74111 提到 29.2）。

---

## 阶段二：第三步（A100 IME2 + VLEN=1024）与 LLM 引擎

### d3574965 feat(riscv): step 3 -- IME2 on A100, Q8 LLM decode engine on RVV
- 说明：IME2 单独编成 `libtl_riscv_ime2.so`；同时搭起 Q8 权重的 LLM 解码引擎。
- 原理：IME2（`smt.vmadot`，`-march=..._xsmtvdotii`）只能跑 cpu8-15，绝不能链进 libtilelang.so；
  引擎每层算子序列是一个 TIR prim_func，`T.call_extern` 调 `llm.h` 里的 RVV 内核。
- 实测：IME2 裸内核 1024³ int8 840 GMAC/s、int32 零误差。

### b0c2d3f3 docs(riscv): implementation report -- steps 1-3, LLM engine, llama.cpp gap analysis
- 说明：三步实现报告 + 与 llama.cpp 的差距账。
- 原理：把"0.6B 差 19%、1.5B 差 2 倍"归因到权重字节数（对方 Q4 比我们 int8 少一半）。
- 实测：Qwen3-0.6B 24.4 tok/s、DS-1.5B 8.4；llama.cpp 30.2 / 16.2。

### 75f6c94d docs(riscv): note that the compiler-bug workaround is maxseq=128 only
- 说明：记录 lowering bug 的绕行只对 maxseq=128 有效。
- 原理：TVM `StorageRewrite` 的 scalable-vector 崩溃在部分形状复现，整模型程序失败→逐层兜底。
- 实测：maxseq=64 也会命中（记录，未根治）。

### 53952045 feat(riscv-llm): 4-bit weight path, sharded checkpoints, model baselines
- 说明：加 4bit 权重路径 + safetensors 分片读取 + 多模型基线。
- 原理：Q4_0 网格（32 值/块）内核数值正确，但 RVV 软件解包让内核从带宽受限变指令受限。
- 实测：0.6B int8 22-24 tok/s，q4 掉到 16.5（负结果，记录）；1.7B 8.54 tok/s。

### c2fc4775 bench(riscv): IME2 decode ceiling at M=16 -- 20.6 tok/s, below the RVV path
- 说明：测 IME2 在 decode 形状（M=16 小矩阵）的上限。
- 原理：小矩阵在 IME2 上只有 9-11 GB/s，搬过去反而比 RVV 慢。
- 实测：M=16 GEMM-only 48.5 ms → 20.6 tok/s 上限，低于 X100 RVV 的 decode 41 ms/token。

### a87a162e docs(riscv): study the SpacemiT IME profile; vmadot.hp verified, and its limit
- 说明：读 SpacemiT IME profile，确认 vmadot.hp 语义与它的边界。
- 原理：vmadot.hp i4 要求激活也 int4；RVV 解包没收益、原生 4bit 点积才是出路。
- 实测：无新数据，是原理性结论（为后面 Q4+IME2 铺路）。

### 99876a3b docs(riscv): close the decode-speed analysis with the measured arithmetic
- 说明：decode 速度账收口。
- 原理：654 MB/token ÷ 17.2 GB/s 流式峰值 = 38 ms = 26 tok/s，是 int8 在这颗芯片的硬上限。
- 实测：GEMV 已到流式峰值的 90%（15.3 / 17.2）。

### ed9c05ac profile(riscv-llm): a decode step is 35.8 ms of layer GEMV + 7 ms lm_head + 3 ms rest
- 说明：把 decode 单 token 拆成三段计时。
- 原理：用 profile_gemv.py 把层程序剥到只剩 7 个 GEMV，三段相减。
- 实测：层 GEMV 35.8 ms（468 MB @ 13.1 GB/s）+ lm_head 7 ms + 其余 3 ms ≈ 45 ms。

### 0b0ac553 bench(riscv): GEMV stream rate does not depend on call granularity
- 说明：测调用粒度是否影响带宽。
- 原理：冷 512 MB 权重，224 次小调用 vs 1 次大调用。
- 实测：224 次 2048×1024 = 15.3 GB/s ≈ 1 次 151936×1024 = 15.2 GB/s（粒度无关）。

### e47d0920 docs(riscv): the remaining optimisation options, all measured
- 说明：把剩余优化方案全部实测并记录结论。
- 原理：大页/软件预取/行内融合 scale/合并调用 全都不改带宽（已在内存墙）。
- 实测：流式读 7 线程 17.2 GB/s、MADV_HUGEPAGE 15.8-16.5、预取 6.9/16.4、融合 15.2。

### 398cd604 feat(riscv-llm): batched prefill (M=8) -- correct, but only 1.15x in situ
- 说明：M=8 批量 prefill，端到端只 1.15×。
- 原理：8 个 e32m4 累加器占满寄存器堆溢出；后续 28b4a282 改 e32m1 才解决。
- 实测：20 token prefill 870 → 761 ms（m4 累加器）。

### c9e74111 perf(riscv): 16-lane columns in the fp32 GEMM template (5.5 -> 29.2 GFLOP/s)
- 说明：GEMM 模板列向量 e32m1 → e32m2。
- 原理：m1 只有 8 通道，256 宽 B 面板要扫 32 遍、2KB 跨步读；m2 减半扫次数。
- 实测：512³ 4 线程 48.5 → 9.2 ms（5.53 → 29.23 GFLOP/s，三次 30.06/30.08/31.29）。

### 28b4a282 perf(riscv-llm): 8-lane accumulators in the M=8 prefill GEMV (3.6 -> 16.0 GB/s)
- 说明：M=8 prefill GEMV 累加器 m4 → m1。
- 原理：8 条 m4 占满 32 向量寄存器；m1（8 通道、每 32 权重切 4 片）腾出 24 个。
- 实测：3.60 → 16.0 GB/s；端到端 prefill 761 → 593-646 ms。
  （⚠ 后经 d920b23b 前后复查，16.0 里有一部分是基准 trip-count 写错虚高，真实约 5 GB/s，
  见 5b5cdc82 的更正。）

### 7000df4b feat(riscv-llm): convert a Q8_0 GGUF straight into the engine blob
- 说明：from_gguf.py——Q8_0 GGUF 直接转引擎 blob，免下 16 GB fp16 权重。
- 原理：Q8_0 已带 32 元素/fp16 scale 网格，8-bit 数据直接拷贝。
- 实测：Qwen3-8B GGUF 8.7 GB → 9.287 GB blob（约 6 分钟）；0.6B 与 HF 导出相关度 0.9999。

### f5b32f88 exp(riscv-llm): K-blocked M=8 GEMV, and why it stays off
- 说明：K 分块的 M=8 GEMV，实验后保留为宽 K 候选但不默认启用。
- 原理：孤立 K=3072 有 3.8×，但原位 prefill 629→842 ms 退化（每块重开 OpenMP region + x 不热）。
- 实测：孤立 1.14 → 4.39 GB/s；原位 629 → 842 ms（回滚）。

### 5e4d5146 docs(embodied): what actually runs on K3, with numbers
- 说明：具身模型在 K3 上的实测清单。
- 原理：（与 TileLang 主线并行，记录 VLA/ACT 等在板上的可行性。）
- 实测：SmolVLA 7.5s（EP）/21.2s（CPU）等，见 `docs/embodied`。

### d920b23b perf(riscv-llm): slice down_proj by K, 629 -> 558 ms prefill
- 说明：down_proj（K=IM）按 1024 切片，每片一次 OpenMP region。
- 原理：整块调用时 8×3072 的激活瓦片 96 KB 超过 L1，每输出行重读 → 1.1 GB/s；切片后 4.9 GB/s。
- 实测：单层 2.80 → 0.64 ms；20 token prefill 629 → 558 ms；decode 23.0 不变。

### 5b5cdc82 docs(riscv): rewrite the backend note with the measured numbers
- 说明：重写 `RISCV_BACKEND.md`，并更正 M=8 内核吞吐（16.4 是基准 trip-count 写错）。
- 原理：变体基准的内层 trip 写成 `vl/VLMAX_EXPR`（m1 下=1），每 32 权重只读 8 个 → 虚高 4×。
- 实测：更正后 N=2048/K=1024 下 m4 3.88、m2 5.89、m1 5.00 GB/s（真实值）。

### fac4d5e7 chore: ignore the build symlinks too
- 说明：.gitignore 加 build/build_riscv（软链接）。
- 原理：`build -> build_riscv` 是软链接，`build/` 模式匹配不到它。
- 实测：git status 变干净。

### acd4265c feat(riscv-llm): Q4_0 weights on the A100 IME2 engine, 23.8 -> 46.4 tok/s
- 说明：Q4_0 + A100 IME2 解码路径（换引擎 + 减半字节）。
- 原理：vendor `ime2::gemm_kernel_i8i4`（4-bit）19-22 GB/s vs 我们的 int8 RVV 15.3，且每权重只读 4.5 bit。
- 实测：0.6B 23.8 → 46.37 tok/s；DS-1.5B 8.4 → 16.34（对照 llama.cpp 53.4 / 16.2）。

### 864a13c0 docs(riscv): Q4/IME2 decode path and the cross-check against the two kubuds branches
- 说明：Q4/IME2 落地文档 + 与 karen/mahiru 两分支的交叉验证。
- 原理：mahiru 的 Q4+IME2 路线复现成功（46.4 vs 其 45.9），karen 的 cblas 76→实测 80.9。
- 实测：见 `docs/CROSSVAL_KAREN_MAHIRU_2026-11-10.txt`；OpenBLAS 线程池由 OPENBLAS_NUM_THREADS 定。

### deb0e261 build(riscv-llm): script for the IME2 Q4 bridge
- 说明：`build_ime2_bridge.sh`——把 vendor spert 内核对象编成 libime2bridge.so。
- 原理：bridge 依赖 spacemit-llama.cpp 已编好的 ime2/rvv/repack 对象 + libspert；补 main_kernel 空桩。
- 实测：产出 159 KB libime2bridge.so，3 个入口（repack/gemv/gemm）。

### d64c750c feat(riscv-llm): M=4/M=8 Q4 IME2 GEMM for prefill, bit-exact, 2.5-3.1x
- 说明：M=4/M=8 批量 Q4 GEMM（prefill 用）。
- 原理：`gemm_kernel_i8i4_m4` 读的是 152 字节交错 A 块（[4×f32][4×i16][4×32 i8]），
  由 `rvv::quantize_a_4row_i8` 生成；M=8 = 两个 4 行组。
- 实测：4096×1024 0.956→0.314（3.05×）、1024×3072 2.48×、151936×1024 3.15×，maxdiff 0.0。

### 785451ae feat(riscv-llm): IME2 M=4 prefill path, 25 -> 12.0 ms/token
- 说明：把 M=4 GEMM 接进引擎（build_layer_m4）。
- 原理：批量线性层走 A100 m4 内核，逐行 attention 保 causal 顺序。
- 实测：84 token prefill 1009 → 732 ms（8.7 ms/token），marginal 7.9 ms/token；decode 不变。

### acb5924d docs(riscv): prefill ledger -- marginal 7.9 ms/token, and why the tail padding was reverted
- 说明：台账 + 记录"尾块补满 chunk"被否掉。
- 原理：同一提示词 ±25% 噪声（板上有别的租户），30 ms 的改动无法用单次分辨。
- 实测：84 token 951/1355/1471 ms；尾巴补满无收益，回滚。

### ea3e3270 perf(riscv-llm): pick the prefill chunk size per prompt, 84 tok 1009 -> 732 ms
- 说明：按提示词余数在 4/8/16 里选 prefill chunk。
- 原理：大 chunk 摊薄逐行开销但浪费余数；编译三份、按余数代价模型选。
- 实测：原位 min MR=4 9.77 / 8 7.78 / 12 7.17 / 16 7.12 ms/token；端到端 84 tok 732 ms。

### 0ed63581 docs(riscv): prefill cost breakdown -- 91% GEMM at pos=0, attention 20% at p=128
- 说明：chunk 成本拆解。
- 原理：同一 chunk 在不同 pos 计时，斜率 = attention。
- 实测：pos=0 110.4 ms（6.9 ms/token，91% GEMM）；attention 0.212 ms/(pos·chunk)，p=128 占 20%。

### 90ef19c1 build(riscv-llm): keep the full-rebuild recipe in-tree
- 说明：把全量重建脚本收进仓。
- 原理：`build_riscv/` 与 `~/.tilelang/cache` 会一夜消失（无重启无 rm，疑似他租户清理），重建脚本才是持久资产。
- 实测：cmake USE_LLVM=OFF + make -j8 ≈ 2.5h。

### a07061f2 build(riscv-llm): pin every Python cmake var to the venv (3.12) in the rebuild
- 说明：重建时把 Python 相关 cmake 变量全部指到 venv 的 3.12。
- 原理：cmake `find_package(Python)` 选到 /usr/bin/python3（3.14，无 Cython、ABI 不符）；
  uv 管的 venv python 自带头文件 + libpython。
- 实测：重建不再报 "No module named cython"。

### 606efbbd build(riscv-llm): USE_ALTERNATIVE_LINKER=OFF, USE_ASCEND=OFF in the rebuild
- 说明：关掉 TVM 的备选链接器与 Ascend。
- 原理：装了 lld（给 IREE）后 TVM `USE_ALTERNATIVE_LINKER=AUTO` 会选中 lld，gcc 目标链接报
  "relocation refers to a symbol in a discarded section"。
- 实测：libtilelang.so（12.6 MB）重建成功；`pytest testing/python/cpu` 86 passed / 2 skipped。

### 75527693 docs(riscv): 8B Q4_0 + IME2 = 4.18 tok/s (2.14x over int8); rebuild recovered
- 说明：8B Q4 blob 跑通 + 记录重建恢复。
- 原理：8B Q4_0 blob（5.50 GB）vs int8（9.29 GB），字节减半。
- 实测：8B decode 1.95 → 4.18 tok/s（llama.cpp Q4_K_M 4.46 的 94%）。

### 402cade7 perf(riscv-llm): one shared 8-worker IME2 stream, 32-aligned column split
- 说明：单共享 8-worker spert 流 + 按 32 块切列。
- 原理：两个 Stream 会超订 8 核（"!!!!"、prefill 30s）；每 worker 按 ceil(N/nw) 切列在奇数 N
  （lm_head 151936/8=18992）留 16 列非对齐 → 越界。
- 实测：prefill MR=16 4 worker 113.9 → 8 worker 89.8 ms（1.27×）；decode 约持平。

### b81dd6db docs(riscv): ledger -- 8-worker IME2 stream, 1.27x prefill, rebuild regression green
- 说明：台账记录本轮 8-worker 结果 + 重建后回归绿。
- 实测：prefill 5.6 ms/token（llama.cpp 3.3）；`pytest` 86 passed / 2 skipped。

### 440f33f5 docs(riscv): M=8 prefill kernel feasibility analysis
- 说明：M=8 内核可行性分析（读 vendor 源码后的结论）。
- 原理：Q4 快路是 RVV `vmadot.hp i4`（`_xsmtvdotii`，MIT）不是 IME2 tensor core；
  m4 的 4×32 tile 占满 32 寄存器，M=8 须收窄到 8×16；M/8 遍 < M/4 遍 → 2×。
- 实测：无（分析文档），预期 prefill 5.6 → 2.8 ms/token。

### 9bfa523d docs(riscv): locate the public XSMTVDotII / vmadot spec trail for the M=8 kernel
- 说明：定位公开规范。
- 原理：X60=XSMTVDot（smt.vmadot，LLVM 22），A100=XSMTVDotII（vmadot.hp i4）；B 须转置成 (N,K)。
- 实测：无（文档），给 M=8 实现指路。

### 8401be05 perf(riscv-llm): 4-tile grid for decode, 8 for prefill, one stream
- 说明：decode 发 4-tile grid、prefill 发 8-tile，共用一 stream。
- 原理：decode 是 DRAM 界、矩阵小，8 worker 只加开销；prefill 有指令余量要 8。
- 实测：1.5B decode grid{8} 13.2 → grid{4} 14.4 tok/s；8B 4.05；prefill 保留 1.27×。

---

## 汇总：关键实测数字

| 项 | 数字 |
|---|---|
| `T.gemm` | 标量 1.22 / RVV 29.2 / cblas 80.9 GFLOP/s（512³） |
| IME2 int8 裸内核 | 840 GMAC/s（1024³，int32 零误差） |
| decode（Q4_0+IME2） | 0.6B 44-46、1.5B 14.4、8B 4.18 tok/s |
| 同板 llama.cpp | 0.6B Q4_0 53.4、1.5B Q4_K_M 16.2、8B Q4_K_M 4.46 |
| prefill | 5.6 ms/token（llama.cpp 3.3；M=8 内核是唯一出路，规范已定位） |
| 回归 | `pytest testing/python/cpu` 86 passed / 2 skipped |
