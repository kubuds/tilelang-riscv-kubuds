"""RISC-V codegen entry points.

Both the host and the device halves of a RISC-V kernel are compiled by TVM's
LLVM code generator (``target.build.llvm``) -- on a RISC-V CPU there is no
separate device, so host and device code share the same triple and the same
code generator.
"""

from __future__ import annotations

from tilelang.backend.device_codegen import global_func_device_codegen

build_llvm = global_func_device_codegen("target.build.llvm")
