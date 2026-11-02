"""Toolchain flags for the RISC-V backend.

The target kind carries ISA facts as attributes (see src/riscv/target.cc);
this module is the single place that turns them into compiler flags.  Keeping
it separate from the pass pipeline means a flag change never re-triggers
layout inference or codegen, only the final host compile.

Flag provenance
---------------
* ``-march=`` / ``-mabi=`` come from the target attributes ``march``/``mabi``.
  Defaults are ``rv64gc``/``lp64d``: the RV64GC baseline that any
  riscv64-linux-gnu gcc accepts, matching SG2042 userland.
* ``-mrvv-vector-bits=`` is emitted only when ``vlen`` is known.  This is
  what makes RVV kernels VLEN-specific on purpose rather than by accident:
  K3 has X100 (VLEN=256) and A100 (VLEN=1024) in one machine, so a kernel
  compiled for one must not be silently reused on the other.
* OpenBLAS is linked, never vendored.  License: BSD-3-Clause, compatible with
  Apache-2.0; we take the system soname so no source is copied into this
  repository.

Everything here is deliberately additive: with ``target="riscv"`` and no
extra attributes, the resulting flags are exactly what the C backend would
produce for a generic riscv64 build.
"""

from __future__ import annotations

import os

from tvm.target import Target

# System soname shipped by libopenblas0 on Bianbu/Debian riscv64.  There is no
# `libopenblas.so` development symlink in the runtime-only package, so the
# linker needs either -l:libopenblas.so.0 or an absolute path.
_DEFAULT_OPENBLAS = "/usr/lib/riscv64-linux-gnu/libopenblas.so.0"


def _attr_str(target: Target, name: str, default: str) -> str:
    value = target.attrs.get(name, default)
    return str(value)


def get_compile_flags(target: Target) -> list[str]:
    """Return extra gcc flags for a ``riscv`` target (empty for others)."""
    if target.kind.name != "riscv":
        return []

    march = _attr_str(target, "march", "rv64gc")
    mabi = _attr_str(target, "mabi", "lp64d")
    flags = [f"-march={march}", f"-mabi={mabi}"]

    # Only pin the vector width when the caller actually knows it.  An unknown
    # VLEN must stay unspecified so the compiler can emit VLEN-agnostic code.
    vlen = int(target.attrs.get("vlen", 0))
    if vlen:
        flags.append(f"-mrvv-vector-bits=zvl{vlen}")

    if bool(target.attrs.get("ime2", False)):
        # Consumed by the IME2 templates added in step 3.
        flags.append("-DTL_RISCV_IME2=1")

    blas = _attr_str(target, "blas", "")
    if blas:
        lib = os.environ.get("TL_RISCV_OPENBLAS", _DEFAULT_OPENBLAS)
        flags.extend([lib, "-lm"])

    return flags
