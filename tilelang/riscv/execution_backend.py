"""Execution backends for the RISC-V target.

Only ``tvm_ffi`` is enabled at step 1.  The Cython adapter is bound to the
``c`` target kind elsewhere (``tilelang/cpu/execution_backend.py``), and
reusing it for ``riscv`` would need a target-kind check that does not yet
exist; tvm_ffi already handles "compile a C source and load it" for any
kDLCPU target, which is exactly what step 1 needs.
"""

from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec

EXECUTION_BACKENDS = [
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
]
