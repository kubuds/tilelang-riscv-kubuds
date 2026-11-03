# IME extension notes (SpacemiT `Zvvm_spacemit`, doc v0.6)

Reference: https://github.com/spacemit-com/docs-ai/blob/main/zh/architecture/ime_extension.md
(consulted 2026-11-03). Everything below was measured on this K3; the doc is
the specification, the numbers are ours.

## What the doc adds over what we already had

Our IME2 kernel used only `smt.vmadot` (int8 -> int32, A100 tile 8x16x8,
A60/X100 tile 4x8x4). The doc describes the full profile:

| sub-extension | input | accumulate | tile (VLEN=1024) |
|---|---|---|---|
| Xsmti8i32mm | Int8 | Int32 | 8x16x8 |
| Xsmti4i32mm | Int4 (both operands) | Int32 | 8x32x8 |
| Xsmti8*16mm_scl16f (`vmadot.hp`, i8) | Int8 + fp16 scale | FP16 | 8x16x8 |
| Xsmti4*16mm_scl16f (`vmadot.hp`, i4) | Int4 + fp16 scale | FP16 | 8x32x8 |
| Xsmt*16fp32mm (`vfwmadot`) | FP16/BF16 | FP32 | 8x8x8 |

The interesting one for LLM decode is `vmadot.hp`: it multiplies the integer
dot product by a per-column fp16 scale taken from `V0`/`V1` (group selected by
`imm3`) and accumulates into the 16-bit destination. That is block
quantisation in hardware -- no software dequant loop.

## Verified on this K3 (A100, cpu8-15)

`testing/riscv/ime/test_vmadot_hp_semantics.c` reproduces the document's own
vector (A[r][k]=r+k, B^T[c][k]=c+k, scale=1..8):

    row 0: 1240 2720 4440 6400 8600 11040 13720 16640   (matches the doc)

Three elements differ by 8 counts (16736 vs 16744). That is not a layout
error: the accumulator is FP16, and above 16384 the ULP is 16, so the
accumulation itself rounds. Consequence for LLM use: fp16 accumulation has to
be flushed to fp32 every few K tiles, otherwise the error grows with K.

`testing/riscv/ime/bench_vmadot_hp_throughput.c`, one thread, registers only:

    vmadot.hp i8   183.8 GMAC/s
    vmadot.hp i4   368.5 GMAC/s

## The catch: i4 hp needs 4-bit activations

The operand types are symmetric: `vmadot.hp ..., i4` is Int4 x Int4, and
`vmadotsu.hp` is Int4 x UInt4. There is no "4-bit weights, 8-bit activation"
variant in this profile. So the 2x weight-density win requires quantising the
activations to 4 bits as well, which measures ~12% RMS per layer on this model
class (see the int4 notes in the skill) -- not usable for a 0.6B model.

With `i8` (8-bit weights + 8-bit activations + hardware scale) the kernel is
compute-rich (183 GMAC/s/thread) but the weight stream is unchanged: decode
stays DRAM-bound at the same ~33 ms/token as the plain `vmadot` path. It
removes the software dequant, not the bytes.

## Toolchain trap worth remembering

`vsetvli x0, x0, e32, m1, ta, ma` (rd = x0) raises SIGILL on the A100 core;
the same instruction with a real destination register (`vsetvli t0, x0, ...`)
works. This cost an hour of chasing "the hp instruction traps" when the trap
was in the vtype setup before it. See `probe_vsetvli_zero_trap.c`.

## What this means for the speed goal

1. Moving decode GEMMs to A100 with the current int8 kernel is a loss:
   small matrices only reach 9-11 GB/s there (see bench_ime2_decode_ceiling).
2. `vmadot.hp` with int8 does not change that: same bytes, and the compute is
   already not the limit.
3. A 4-bit decode path must either accept 4-bit activations (accuracy risk) or
   stay in software on X100. Our software Q4 GEMV currently measures 16.5 tok/s
   against 22-24 for int8 because unpacking costs 13 vector instructions per 32
   weights; cutting that to ~9 (fold the -8 offset into one correction term per
   block) is the cheapest remaining lever, worth ~1.3-1.4x.
