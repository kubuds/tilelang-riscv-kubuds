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
from pathlib import Path

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

    # GCC has no numeric form for -mrvv-vector-bits: it accepts only
    # "scalable" or "zvl", and the concrete width comes from the Zvl*b
    # extension in -march.  (Clang would take =zvl256, GCC does not.)  Pinning
    # the width is what keeps an X100 kernel (VLEN=256) from being silently
    # reused on an A100 (VLEN=1024).
    vlen = int(target.attrs.get("vlen", 0))
    if vlen and f"zvl{vlen}b" not in march:
        march = f"{march}_zvl{vlen}b"
    # tl_templates/ lives under <repo>/src; the generated C includes it by
    # relative path, so the host compile must see it.  The CPU backend gets
    # this from its own include list; the riscv kind is a new kind, so it has
    # to spell it out.
    src_inc = Path(__file__).resolve().parents[2] / "src"
    flags = [f"-I{src_inc}", f"-march={march}", f"-mabi={mabi}"]
    # The default -O2 leaves the TIR loop nest scalar (measured: zero vector
    # instructions at rv64gcv).  -O3 enables the loop vectoriser; -ffast-math
    # is deliberately NOT used because it would change numerics.
    flags += ["-O3", "-ftree-vectorize"]
    # The device translation unit (lib0.c) does not include the templates
    # header -- only the host unit does -- so vector types such as float4 are
    # undeclared there.  -include forces the header into every TU.  The
    # riscv.h entry header pulls in the generic cpp/common.h and adds the
    # RVV primitives, so kernels can call tl_riscv_* from any lowering.
    flags += ["-include", "tl_templates/riscv/riscv.h"]

    # Kernels in tl_templates/riscv use `#pragma omp parallel for`; without
    # -fopenmp those pragmas are silently ignored and every GEMV/GEMM runs on
    # one thread (measured 6-8x slower on the LLM decode path).
    flags.append("-fopenmp")

    # Only pin the vector width when the caller actually knows it.  An unknown
    # VLEN must stay unspecified so the compiler can emit VLEN-agnostic code.
    if vlen:
        flags.append("-mrvv-vector-bits=zvl")

    if bool(target.attrs.get("ime2", False)):
        # Consumed by the IME2 templates added in step 3.
        flags.append("-DTL_RISCV_IME2=1")

    blas = _attr_str(target, "blas", "")
    if blas:
        lib = os.environ.get("TL_RISCV_OPENBLAS", _DEFAULT_OPENBLAS)
        # The generated C only includes tl_templates/cpp/common.h, so cblas.h
        # must be force-included; -include is order-independent and works for
        # both .c and .cc compilation units.
        # TVM compiles the generated C with g++, so the CBLAS enums in
        # cblas.h are typed and int literals from call_extern are rejected;
        # -fpermissive keeps the (correct) integer values usable.
        # TVM puts options before the object files, so --as-needed would
        # discard the library before it sees the reference.
        flags.extend(["-fpermissive", "-Wl,--no-as-needed", lib, "-lm"])

    return flags
