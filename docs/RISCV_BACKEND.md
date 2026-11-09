# RISC-V backend (TileLang v0.1.15)

A `riscv` target kind for TileLang, covering three steps on the SpacemiT K3
(8x X100 with RVV VLEN=256 + 8x A100 with IME2):

1. plain RV64GC — TVM's CPU path, verified as the baseline;
2. RVV 1.0 on the X100 cluster — an in-tree fp32 GEMM for `T.gemm` plus
   RVV primitives for the LLM path;
3. A100 — IME2 int8 GEMM (separate `.so`), RVV at VLEN=1024.

On top of that sits a small int8 LLM engine that runs Qwen3-0.6B/1.7B/8B and
DeepSeek-R1-Distill-Qwen-1.5B end to end on the board.

```
target = {"kind": "riscv", "march": "rv64gcv", "mabi": "lp64d",
          "vlen": 256,            # 256 X100 / 1024 A100 / 0 = unknown
          "rvv": True,            # in-tree RVV GEMM for T.gemm
          "blas": "openblas",     # optional, wins on throughput
          "ime2": True}           # A100 int8 GEMM (separate .so)
```

## 1. Baseline RV64GC

`test_riscv_step1.py` (512^3 fp32, hand-written TIR loop nest): 1052 ms,
2.8e-07 max rel error vs torch, `objdump | grep -c vsetvli` = 0. The C target is
reused verbatim, only `-march=rv64gc` changes, so the baseline is "TVM's CPU
path on RV64GC" by construction. Nothing in the backend emits ISA-specific code
unless `rvv`/`ime2`/`vlen` ask for it, so an SG2042 userland gets the same
binary shape (SG2042 was not reachable from this machine; only K3 was measured).

## 2. RVV 1.0 (VLEN=256, X100)

`T.gemm` had no CPU implementation at all in v0.1.15 (`SelectInst` returns
`cpu.scalar`, nothing registers it). `tilelang/riscv/gemm_riscv.py` supplies the
first one with three lowerings.

512^3 fp32, best-of-three, one interleaved run:

| lowering | threads | time | throughput |
|---|---:|---:|---:|
| scalar TIR nest | 8 | 219.5 ms | 1.22 GFLOP/s |
| `rvv=True`, e32m1 columns (first cut) | 8 | 48.5 ms | 5.53 GFLOP/s |
| `rvv=True`, e32m2 columns | 4 | 9.2 ms | **29.2 GFLOP/s** |
| `blas="openblas"` | 4 | 6.6 ms | 40.4 GFLOP/s |
| `blas="openblas"` | 8 | 5.2 ms | 51.9 GFLOP/s |

OpenBLAS is linked, not vendored (BSD-3-Clause, compatible with Apache-2.0).

What the numbers cost to find (`src/tl_templates/riscv/riscv.h`):

* one accumulator chain per row is `vfmacc` latency-bound: 4.4 GFLOP/s per core;
  MR=8 independent chains turn it into an issue-bound loop.
* the column vector must be `e32m2`, not `e32m1`. With m1 (8 lanes at VLEN=256)
  a 256-wide B panel is swept 32 times per k step and every load walks B with a
  2 KB stride; m2 halves the sweeps. This is the whole 5.5 -> 29.2 jump.
* thread count is workload-shaped: pure FMA scales 63/122/151/92 GFLOP/s at
  1/2/4/8 threads, so compute-bound kernels want **4** and memory-bound GEMV
  wants **7** (8 collapses to ~5 GFLOP/s on the GEMM).
* `-march=rv64gcv` + `-O3` alone does *not* vectorise the TIR loop nest (12
  vector instructions, all prologue). Real RVV has to be written, not requested.

The decode path does not use `T.gemm` at all: a 1-token GEMV is DRAM-bound, so
`llm.h` implements Q8 GEMV / RMSNorm / RoPE / GQA attention / SiLU directly with
RVV intrinsics. Same source compiles for VLEN=256 and VLEN=1024 (`vsetvl`
everywhere, verified on A100).

## 3. A100: IME2 and VLEN=1024

`testing/riscv/test_riscv_ime2.py` moves the process to cpu8-15
(`/proc/set_ai_thread`) and checks int8 GEMM against a numpy reference:

| M=N=K | max abs err | bare kernel |
|---|---:|---:|
| 256 | 0 | 124 GMAC/s |
| 512 | 0 | 372 GMAC/s |
| 1024 | 0 | **840 GMAC/s** |

Exact (int32) match. The number is the *bare* kernel: packing is a separate,
load-time step. With packing folded into the timed call the same 1024^3 problem
reads 38 GMAC/s, which is the packer, not the matrix unit. The kernel is
compiled into `libtl_riscv_ime2.so` with `-march=..._xsmtvdotii`, deliberately
not linked into `libtilelang.so` (the extension must not leak into the RVV
fallback, and it only executes on cpu8-15).

RVV at VLEN=1024 is correct on A100 (gemv 256x1024, max err 4.2e-04) but ~25x
slower than the same code on X100; A100 is for IME2, X100 for everything else.

## 4. LLM engine

`tilelang/riscv/llm/engine.py` + `tools/riscv_llm/{export_weights,from_gguf,run_model}.py`.

* Weights: int8, one fp16 scale per 32-element block (the llama.cpp Q8_0 grid),
  stored planar (all q, then all scales) so the GEMV inner loop reads 32 int8 +
  one half instead of interleaved metadata. `export_weights.py` reads
  safetensors (sharded checkpoints supported); `from_gguf.py` converts a Q8_0
  GGUF directly, which skips the 16 GB fp16 download for 8B models.
* Schedule: one layer's operator sequence is a single TIR prim_func; Python only
  loops over layers and the head. When the whole model lowers it is one program
  and one FFI call per token; otherwise the engine falls back to per-layer
  programs (36 FFI calls per token for 8B).
* Per-op C kernels live in `src/tl_templates/riscv/llm.h`, the same split
  TileLang uses on CUDA (`T.gemm` -> CUTLASS templates).

Measured (K3, OMP_NUM_THREADS=7, greedy, 128-token cache):

| model | weights | prefill | decode | same-board llama.cpp |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 0.94 GB | 25 ms/token (batched) | **22-25 tok/s** | 30.2 (Q8_0) |
| Qwen3-1.7B | 2.45 GB | 120 ms/token | 8.54 tok/s | 15.73 (Q4_K_M) |
| DeepSeek-R1-Distill-1.5B | 2.13 GB | 120-224 ms/token | 8.5 tok/s | 16.22 (Q4_K_M) |
| Qwen3-8B | 9.29 GB | 533 ms/token | 1.95 tok/s | 4.46 (Q4_K_M) |

All four produce coherent continuations; the tokenizer and prompts are shared
with the llama.cpp builds.

Decode is at the memory wall. A 0.6B token is 654 MB of weights, the board's
measured streaming read peak is 17.2 GB/s, and the GEMV sustains 15.3 GB/s
(90%); the arithmetic floor is 38 ms/token = 26 tok/s. The vendor's 30.2 tok/s
implies either fewer bytes per token or a different measurement scope.

Batched prefill (`build_layer_m8`) processes 8 prompt tokens per weight stream.
It is 1.4x faster than per-token prefill at H=1024 and **5.5x slower at H=4096**,
so the engine enables it only when `H <= 2048`.

down_proj (K = IM) is sliced into 1024-element calls. A whole-K call re-walks an
8xK activation tile (96 KB at 0.6B, 393 KB at 8B) once per output row and drops
to 1.1 GB/s; slicing keeps each tile at 32 KB in L1 and costs one OpenMP region
per call: 2.80 -> 0.64 ms per layer, 629 -> 558 ms for a 20-token prefill.

## 5. Compiler fixes found on the way

1. `src/tl_templates/cpp/common.h`: `float2/float4` were declared outside
   `namespace tl` while `vec_type` lives inside it; every vectorised copy failed
   to compile.
2. `toolchain.py`: the generated *device* TU never included the template header;
   `-include tl_templates/riscv/riscv.h` fixes every kernel at once.
3. `src/transform/storage_rewrite.cc`: `with_lanes(me->coeff)` with `coeff == 0`
   (unknown modular set) creates a *scalable* vector dtype, and the next
   `lanes()` throws "Can't fetch the lanes of a scalable vector". Guarded. This
   is an upstream TVM/tilelang bug, not RISC-V-specific.
4. Known open bug: the same pass still throws for TIR containing a layer loop
   over L>=2 at some shapes (H=1536/IM=8960, and H=4096 for the 8B model), hence
   the per-layer fallback in `engine.py`. Worked around at the schedule level.

## 6. What did not work

Each of these was implemented and measured, then reverted; they are recorded so
nobody repeats them.

| attempt | result |
|---|---|
| 4-bit weights with software unpack | 16.5 tok/s vs 22-24 for int8 (13 vector ops per 32 weights vs 5) |
| IME2 for the decode GEMM | M=16 ceiling 20.6 tok/s, below the X100 RVV path |
| `vmadot.hp` int4 | requires 4-bit activations; accuracy unusable at 0.6B |
| `MADV_HUGEPAGE` on the weight blob | 22.9 vs 23.5-24.4 tok/s, within noise |
| software prefetch in the GEMV | 6.9 GB/s without THP, 16.4 with; no gain over baseline |
| fusing scales into the weight row | 15.2 vs 15.3 GB/s, no change |
| merging QKV into one kernel call | 224 small calls 15.3 vs 1 large call 15.2 GB/s |
| in-kernel K-blocking of the M=8 GEMV | microbench 1.14 -> 4.39 GB/s, but end-to-end prefill 629 -> 842 ms (6 OpenMP regions per call) |

The M=8 kernel itself is cache/issue bound at ~5 GB/s, not bandwidth bound; a
variant benchmark that computed its inner trip count as `vl/VLMAX_EXPR` read a
quarter of each block and reported 16.4 GB/s. Fixed in `gemv_m8_variants.c`.

## 7. Reproduce

```
export PYTHONPATH=$PWD

python bench_riscv_gemm.py 512                 # scalar / rvv / openblas
python testing/riscv/test_riscv_ime2.py        # A100, moves process to cpu8-15

# weights: from safetensors, or straight from a Q8_0 GGUF
python tools/riscv_llm/export_weights.py <hf_dir> /tmp/qwen3.bin
python tools/riscv_llm/from_gguf.py Qwen3-8B-Q8_0.gguf /tmp/qwen3_8b.bin

OMP_NUM_THREADS=7 python tools/riscv_llm/run_model.py /tmp/qwen3.bin \
  <tokenizer.json> --prompt "The capital of France is" --n 64 --raw
```

Regression: `pytest testing/python/cpu -q` -> 86 passed, 2 skipped.

## 8. Licence

All new code is Apache-2.0, same as the repository. OpenBLAS is linked
(BSD-3-Clause). The IME2 kernel is this project's own earlier code
(`tl_k3_opt/src/ime2_own_gemm_v15_ldc.c`), moved in-tree; no third-party source
was copied. References consulted: RISC-V V 1.0 spec, SpacemiT K3 paper
(VLEN/IME2 semantics), llama.cpp MIT (Q8_0 block layout only, no code).
