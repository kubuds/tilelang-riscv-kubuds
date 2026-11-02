"""RISC-V TIR lowering pipeline.

The RV64GC baseline is semantically identical to the generic CPU pipeline:
thread bindings become serial grid loops and the LLVM code generator emits
scalar machine code for the RISC-V triple.  The pipeline is therefore wrapped
around :func:`tilelang.cpu.pipeline.CPUPassPipelineBody` with RISC-V specific
stages inserted where they matter (RVV legalization and the IME matrix-engine
lowering land here in later commits).
"""

from __future__ import annotations

from tvm import IRModule
from tvm.target import Target

from tilelang.cpu.pipeline import CPUPassPipelineBody

from .target import target_is_riscv


def RISCPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    if not target_is_riscv(target):
        raise ValueError(f"RISC-V pipeline received a non-RISC-V target: {target}")
    # Baseline: the CPU pipeline already produces portable scalar TIR.  Future
    # RISC-V specific passes (RVV intrinsics legalization, IME-2 lowering) hook
    # in here.
    return CPUPassPipelineBody(mod, target)
