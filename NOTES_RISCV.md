# RISC-V backend notes (step 2 blockers)

## Correction to commit 4b1f7e56

That commit reported "195x" for cblas over the TIR loop nest. The ratio is
right for the kernel it measured, but that kernel is not a fair baseline:

    bench_riscv_blas.py   T.Kernel(1), 512x512 locals   1014 ms
    test_riscv_step1.py   T.Kernel(4,4), 128^3 tiles     231 ms
    cblas_sgemm (either)                                   5.2 ms

512x512 local arrays spill (1 MB of locals), which is why the single-block
kernel is 4.4x slower. Against the tiled baseline the honest figure is

    231 ms / 5.2 ms = 44x   (0.26 -> 51.75 GFLOP/s compares against the
                             spilling kernel and overstates it at 195x)

Quote 44x unless the naive kernel is explicitly the subject.

## Verified

RV64GC baseline (commit 739bbb1e): 512^3 fp32, max rel 2.77e-07 vs torch,
zero vector instructions, 1052 ms.

rv64gcv_zvl256b + -O3 (commit 5cdf1071): 226-230 ms, but objdump shows only
12 vector instructions, all prologue save/restore. The 4.6x comes from
unrolling, not from RVV.

The 0.0 error is real, not a test bug: torch's matmul on this board produces
bit-identical results to our loop kernel (0/262144 elements differ), i.e. it
is a scalar reference, not a BLAS one. Consequence: once cblas is used,
correctness must be judged by tolerance, never bitwise.

## Why cblas_sgemm does not compile through T.call_extern

TileLang emits its own prototype for every extern symbol:

    int32_t cblas_sgemm(void*, void*, void*, int32_t, ...);

against cblas.h's

    void cblas_sgemm(CBLAS_LAYOUT, CBLAS_TRANSPOSE, ..., const float*, ...);

which is a "conflicting declaration of C function". So the problem is not the
enum conversion (that was only the first symptom) but that the generated
prototype is guessed from DSL argument types and can never match a real C
header.

## Fixes, in order of preference

1. A thin shim owned by this repository, declared in src/tl_templates/riscv/:

       void tl_cblas_sgemm_i(int layout, int ta, int tb, int M, int N, int K,
                             float alpha, float* A, int lda, float* B, int ldb,
                             float beta, float* C, int ldc);

   Its prototype matches what the codegen guesses (int / float*), so it links
   cleanly, and inside it calls cblas_sgemm with proper casts. No code is
   copied from OpenBLAS (BSD-3-Clause, linked not vendored).

2. Teach codegen_c to suppress the auto prototype for symbols the backend
   lists, and to emit C-style casts for enum-typed arguments. More invasive,
   but the right long-term shape if more C libraries are called.

Do (1) first: it unblocks measurement, and the number decides whether the
lowering pass is worth building.

## Ruled out

Compiling the generated unit with gcc instead of g++: TVM's host glue
includes <utility>, so the unit is C++ regardless of which file is .c.
