# RISC-V backend (TileLang v0.1.15)

Status: steps 1-3 implemented and measured on K3 (X100 + A100), plus a working
LLM decode engine that runs Qwen3-0.6B and DeepSeek-R1-Distill-Qwen-1.5B.

```
target = {"kind": "riscv", "march": "rv64gcv", "mabi": "lp64d",
          "vlen": 256,            # 256 X100 / 1024 A100 / 0 = unknown
          "rvv": True,            # in-tree RVV GEMM for T.gemm
          "blas": "openblas",     # optional, wins on throughput
          "ime2": True}           # A100 int8 GEMM (separate .so)
```

## 1. Baseline RV64GC

`test_riscv_step1.py` (512^3 fp32, hand-written TIR loop nest): 1052 ms,
2.8e-07 max rel error vs torch, `objdump | grep -c vsetvli` = 0.  The C target
is reused verbatim, only `-march=rv64gc` changes, so the baseline is
"TVM's CPU path on RV64GC" by construction.  SG2042 was not reachable from
this machine; the same flags (`rv64gc`, `lp64d`, `-O3`) are what an SG2042
userland gcc uses, and nothing in the backend emits ISA-specific code unless
`rvv`/`ime2`/`vlen` ask for it.

## 2. RVV 1.0 (VLEN=256, X100)

`T.gemm` had no CPU implementation at all in v0.1.15 (`SelectInst` returns
`cpu.scalar`, nothing registers it).  `tilelang/riscv/gemm_riscv.py` supplies
the first one with three lowerings:

| lowering | 512^3 fp32, 8 threads | max abs |
|---|---:|---|
| scalar TIR nest | 223 ms / 1.20 GFLOP/s | 0 |
| `rvv=True` (`tl_riscv_gemm_f32`) | 48.5 ms / 5.53 GFLOP/s | 1.2e-04 |
| `blas="openblas"` | 5.2 ms / 51.9 GFLOP/s | 9.9e-05 |

OpenBLAS is linked, not vendored (BSD-3-Clause, compatible with Apache-2.0).

Kernel notes that cost real time to find (`src/tl_templates/riscv/riscv.h`):

* one accumulator chain per row is `vfmacc` latency-bound: 4.4 GFLOP/s per
  core.  MR=8 independent chains: 16.7 GFLOP/s per core (standalone bench).
* cross-core scaling on this board is noisy: the same 1024^3 kernel measured
  2.75 GFLOP/s at 1 thread / 14.9 at 8 in one run, and 10.5 / 7.2 in another.
  Quote ratios inside one interleaved run only.
* `-march=rv64gcv` + `-O3` alone does *not* vectorise the TIR loop nest
  (12 vector instructions, all prologue): the 4.6x over scalar came from
  unrolling.  Real RVV has to be written, not requested.

The decode path does not use `T.gemm` at all: a 1-token GEMV is DRAM-bound, so
`llm.h` implements Q8 GEMV / RMSNorm / RoPE / GQA attention / SiLU directly
with RVV intrinsics.  Same source compiles for VLEN=256 and VLEN=1024
(`vsetvl` everywhere, verified on A100).

## 3. A100: IME2 and VLEN=1024

`testing/riscv/test_riscv_ime2.py` moves the process to cpu8-15
(`/proc/set_ai_thread`) and checks int8 GEMM against a numpy reference:

| M=N=K | max abs err | bare kernel |
|---|---:|---:|
| 256 | 0 | 124 GMAC/s |
| 512 | 0 | 372 GMAC/s |
| 1024 | 0 | 840 GMAC/s |

Exact (int32) match.  The number is the *bare* kernel: packing is a separate,
load-time step (`tl_riscv_ime2_pack_b` / `_pack_a`).  With packing folded into
the timed call the same 1024^3 problem reads 38 GMAC/s, which is the packer,
not the matrix unit.  The kernel is compiled into `libtl_riscv_ime2.so` with
`-march=..._xsmtvdotii`, deliberately not linked into `libtilelang.so`.

RVV at VLEN=1024 is correct on A100 (gemv 256x1024, max err 4.2e-04) but ~25x
slower than the same code on X100; A100 is for IME2, X100 for everything else.

## 4. LLM engine

`tilelang/riscv/llm/engine.py` + `tools/riscv_llm/{export_weights,run_model}.py`.

* Weights: int8, one fp16 scale per 32-element block (Q8_0 grid, planar q/s),
  exported from safetensors by `export_weights.py`.
* Schedule: the operator sequence of one layer is a single TIR prim_func; the
  Python side only loops over layers and the head.  When the whole model
  lowers, it is one program and one FFI call per token.
* Per-op C kernels live in `src/tl_templates/riscv/llm.h`, the same split
  TileLang uses on CUDA (T.gemm -> CUTLASS templates).

Measured (K3, OMP_NUM_THREADS=7, greedy, 128-token cache):

| model | weights | prefill | decode |
|---|---:|---:|---:|
| Qwen3-0.6B | 0.94 GB | ~45 ms/token | 21.7-24.2 tok/s |
| DeepSeek-R1-Distill-Qwen-1.5B | 2.13 GB | ~120 ms/token | 8.4 tok/s |

Both produce coherent continuations.  Threads matter and 8 is worse than 7 on
X100 (OpenMP spin contention): 1/4/7/8 threads on the 2048x1024 GEMV measure
3.0 / 11.9 / 20.4 / 2.7 GB/s.  At 7 threads the 0.6B decode stream (654 MB per
token) runs at ~15 GB/s effective against a ~21 GB/s DRAM roof, i.e. the
kernel is at the memory wall; more speed needs fewer bytes (4-bit weights),
not more compute.

The 1.5B model runs at ~18 GB/s, also at the roof, but its int8 weights are
1.9x the size of llama.cpp's Q4_K_M, so 4-bit is the only remaining lever
there.  A first-cut 4-bit path is not in this commit.

## 5. Compiler fixes found on the way

1. `src/tl_templates/cpp/common.h`: `float2/float4` were declared outside
   `namespace tl` while `vec_type` lives inside it; every vectorised copy
   failed to compile.
2. `toolchain.py`: the generated *device* TU never included the template
   header; `-include tl_templates/riscv/riscv.h` fixes every kernel at once.
3. `src/transform/storage_rewrite.cc`: `with_lanes(me->coeff)` with
   `coeff == 0` (unknown modular set) creates a *scalable* vector dtype, and
   the next `lanes()` throws "Can't fetch the lanes of a scalable vector".
   Guarded.  This is an upstream TVM/tilelang bug, not RISC-V-specific.
4. Known open bug: at the 1.5B shapes the same pass still throws when the TIR
   contains a layer loop over L>=2 (L=1 lowers fine), hence the per-layer
   fallback in `engine.py`.  Reproducer: `try2.py`-style program with H=1536,
   IM=8960, L=2.  Not fixed here; worked around at the schedule level.

## 6. Reproduce

```
cd /work/kubuds/hermes/tilelang_riscv
export PYTHONPATH=$PWD

python bench_riscv_gemm.py 512                 # scalar / rvv / openblas
python testing/riscv/test_riscv_ime2.py        # A100, moves process to cpu8-15

python tools/riscv_llm/export_weights.py \
  /work/kubuds/hermes/models/qwen3_06b_hf /tmp/qwen3.bin
OMP_NUM_THREADS=7 python tools/riscv_llm/run_model.py /tmp/qwen3.bin \
  /work/kubuds/hermes/models/qwen3_06b_hf/tokenizer.json \
  --prompt "The capital of France is" --n 64 --raw
```

## 7. Licence

All new code is Apache-2.0, same as the repository.  OpenBLAS is linked
(BSD-3-Clause).  The IME2 kernel is this project's own earlier code
(`tl_k3_opt/src/ime2_own_gemm_v15_ldc.c`), moved in-tree; no third-party
source was copied.  References consulted: RISC-V V 1.0 spec, SpacemiT K3
paper (VLEN/IME2 semantics), llama.cpp MIT (Q8_0 block layout only, no code).
