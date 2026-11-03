# tile-level T.gemm has no CPU implementation in v0.1.15

Decisive experiment (probe_tgemm.py), same kernel on both targets:

  target "riscv":  ValueError: No GEMM implementation registered for
                   instruction cpu.scalar and target {... "march":"rv64gc" ...}
  target "c":      InternalError: Check failed: layout_map.count(buffer) != 0
                   (0 vs. 0) : The layout for fragment C_l can not be inferred.

So the failure is not riscv-specific and not caused by the new backend: upstream
v0.1.15 has **no registered GEMM implementation for the CPU family at all**.
`register_gemm_impl` is called only by the CUDA backends; `cpu.scalar` is a
string that SelectInst returns but nothing implements.

This is consistent with the upstream CPU test
(testing/python/cpu/test_tilelang_cpu_gemm.py), which hand-writes the triple
loop instead of using T.gemm: on the CPU path, T.gemm is simply not supported
in this version.

## What this changes

"Make T.gemm call OpenBLAS on riscv" is not an optimisation.  It would be the
**first** GEMM implementation for the CPU family, and it must supply both:

  * infer_layout()  -- the fragment layout for C, which is what the "c" target
                       failure above is missing;
  * lower()         -- the body: cblas_sgemm for riscv+blas, loops otherwise.

This also explains the shape of the surrounding ecosystem: RuyiAI's
tilelang-riscv carries ~2553 lines of linalg codegen because it has to provide
this layer for RISC-V, and TongmingLAIC took the TPU/PPL route instead of the
CPU one.

## Where to implement

tilelang/tileop/gemm/registry.py: register_gemm_impl(name, inst_name, predicate,
impl_class).  The impl_class needs infer_layout(target, thread_nums) and
lower(layout_map, target, thread_bounds, thread_index, mbar_phase_expr).
The instruction key comes from C++ SelectInst, which currently returns
"cpu.scalar" for anything that is kDLCPU -- so riscv and c both land there.
A riscv+blas branch can key off that string and dispatch inside the impl.

## Status before this work

Step 1 (RV64GC) is complete and verified for hand-written kernels.
Step 2 proved OpenBLAS is linkable and fast (5.2 ms / 51.75 GFLOP/s, 44x over
a tiled scalar kernel) through a hand-written T.call_extern.  It did not reach
T.gemm, because T.gemm does not exist on this path yet.
