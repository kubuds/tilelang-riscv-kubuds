"""RISC-V device codegen entry points.

Step 1 reuses the C host codegen verbatim: TIR -> C, compiled by the host
gcc with ``-march=<target.march>``.  Keeping this indirection means later
steps can point ``DeviceCodegen`` at an RVV- or IME2-specific builder without
changing backend.py.
"""

from __future__ import annotations

from tilelang.backend.device_codegen import global_func_device_codegen

# Shared with the CPU backend (src/backend/common/codegen/codegen_c_host.cc).
build_c = global_func_device_codegen("target.build.tilelang_c")
