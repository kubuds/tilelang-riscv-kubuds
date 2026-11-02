"""RISC-V backend manifest.

Registers the ``riscv`` backend for every ``llvm`` target whose triple targets
64-bit RISC-V.  Non-RISC-V ``llvm`` targets keep routing through the generic
CPU backend (see :mod:`tilelang.cpu.backend`, whose ``supports_target``
predicate excludes RISC-V triples so the registry can disambiguate).
"""

from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.module import BackendModule, register_backend
from tilelang.backend.pass_pipeline import PassPipeline

from . import codegen, execution_backend, pipeline
from .target import target_is_riscv

BACKEND = register_backend(
    BackendModule(
        name="riscv",
        target_kinds=("llvm",),
        supports_target=target_is_riscv,
        pipelines={
            "llvm": PassPipeline("llvm", pipeline.RISCPassPipelineBody),
        },
        device_codegens={
            "llvm": DeviceCodegen(
                "llvm",
                build=codegen.build_llvm,
                build_without_compile=codegen.build_llvm,
            ),
        },
        host_codegens=STANDARD_HOST_CODEGENS,
        execution_backends=execution_backend.EXECUTION_BACKENDS,
    )
)
