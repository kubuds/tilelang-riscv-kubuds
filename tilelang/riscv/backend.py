"""RISC-V backend manifest.

Design note
-----------
The RISC-V backend deliberately reuses the CPU pass pipeline and the C host
codegen instead of forking them.  Everything that is *not* ISA-specific
(layout inference, tile op lowering, thread binding) is identical to the CPU
path, and duplicating it would create a second maintenance surface that drifts
from upstream.  What is RISC-V-specific is confined to:

  * the target kind and its attributes  (src/riscv/target.cc),
  * the toolchain flags                 (tilelang/riscv/toolchain.py),
  * later: RVV / IME2 operator lowering (src/riscv/codegen/).

This is also why ``device_codegens`` points at the generic ``target.build.
tilelang_c`` function rather than a new one: step 1 must produce *identical*
code to the CPU backend, only compiled for a different -march.  Any later
step that changes generated code should be introduced as a distinct codegen
entry, not by mutating the shared one.
"""

from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.module import BackendModule, register_backend
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.cpu.pipeline import CPUPassPipelineBody

from . import codegen, execution_backend

BACKEND = register_backend(
    BackendModule(
        name="riscv",
        target_kinds=("riscv",),
        pipelines={
            "riscv": PassPipeline("riscv", CPUPassPipelineBody),
        },
        device_codegens={
            # Step 1: byte-identical C source to the CPU backend.
            "riscv": DeviceCodegen("riscv", build_without_compile=codegen.build_c),
        },
        host_codegens=STANDARD_HOST_CODEGENS,
        execution_backends=execution_backend.EXECUTION_BACKENDS,
    )
)
