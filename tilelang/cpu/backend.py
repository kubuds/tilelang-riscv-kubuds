"""CPU backend manifest."""

from tvm.target import Target

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline

def _supports_target(target: Target) -> bool:
    """Own every CPU target except RISC-V.

    RISC-V ``llvm`` targets are served by :mod:`tilelang.riscv.backend`; the
    backend registry requires both parties to declare a predicate once they
    share the ``llvm`` target kind, and the predicate is what keeps the two
    backends mutually exclusive.
    """

    if target.kind.name != "llvm":
        return True
    from tilelang.riscv.target import target_is_riscv

    return not target_is_riscv(target)


BACKEND = register_backend(
    BackendModule(
        name="cpu",
        target_kinds=("c", "llvm"),
        supports_target=_supports_target,
        pipelines={
            "c": PassPipeline("c", pipeline.CPUPassPipelineBody),
            "llvm": PassPipeline("llvm", pipeline.CPUPassPipelineBody),
        },
        device_codegens={
            "c": DeviceCodegen("c", build_without_compile=codegen.build_c),
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
