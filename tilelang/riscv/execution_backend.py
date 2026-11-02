"""Execution backend policy for the RISC-V backend.

Kernels execute through the TVM-FFI runtime (packed-function ABI), exactly like
the generic CPU backend.  Cross-compiled kernels cannot be executed in the
build process; they are exported with :mod:`tilelang.riscv.export` and run on
the target (natively or under QEMU user-mode emulation).
"""

from __future__ import annotations

from tvm.target import Target

from tilelang.backend.execution_backend import ExecutionBackendSpec


def _is_riscv_llvm_target(target: Target) -> bool:
    from .target import target_is_riscv

    return target_is_riscv(target)


EXECUTION_BACKENDS = [
    ExecutionBackendSpec("tvm_ffi", supports_target=_is_riscv_llvm_target, enable_host_codegen=True, enable_device_compile=True),
]
