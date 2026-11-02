"""RISC-V cross-link/export helpers.

TVM's LLVM code generator can already emit RISC-V relocatable objects on any
host (that is the beauty of a retargetable backend), but ``export_library``
links those objects with the *host* C compiler by default, which only works
when the build machine is itself a RISC-V box.  This module provides the
missing piece: a ``fcompile`` for :meth:`tvm.runtime.Module.export_library`
that links RISC-V objects into a RISC-V shared library using a cross toolchain.

Toolchain discovery order:

1. ``TILELANG_RISCV_GCC`` -- explicit path to a riscv64 gcc driver,
2. ``riscv64-linux-gnu-gcc`` / ``riscv64-linux-gnu-gcc-<ver>`` on ``PATH``,
3. ``clang`` with ``--target=riscv64-unknown-linux-gnu`` (when available).

``TILELANG_RISCV_SYSROOT`` can point at a riscv64 sysroot and
``TILELANG_RISCV_CFLAGS``/``TILELANG_RISCV_LDFLAGS`` add extra flags.

On a natively running RISC-V host nothing here is needed: the regular
``tilelang.compile`` path builds and executes kernels in-process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence

from tilelang.backend.target import TargetLike

from .target import RISCV64_TRIPLE, target_is_riscv

try:
    from tvm.contrib import cc as _tvm_cc
except Exception:  # pragma: no cover - runtime-only environments
    _tvm_cc = None


def _split_env_flags(var: str) -> list[str]:
    value = os.environ.get(var, "")
    return [item for item in value.split() if item]


def find_riscv_cc() -> str | None:
    """Locate a C compiler driver able to produce riscv64-linux-gnu objects."""
    explicit = os.environ.get("TILELANG_RISCV_GCC")
    if explicit:
        return explicit
    for name in ("riscv64-linux-gnu-gcc", "riscv64-linux-gnu-gcc-14", "riscv64-linux-gnu-gcc-13", "riscv64-linux-gnu-gcc-12"):
        found = shutil.which(name)
        if found:
            return found
    return None


def riscv_link_options() -> list[str]:
    """Extra link options derived from the environment."""
    options: list[str] = []
    sysroot = os.environ.get("TILELANG_RISCV_SYSROOT")
    if sysroot:
        options.append(f"--sysroot={sysroot}")
    options += _split_env_flags("TILELANG_RISCV_CFLAGS")
    options += _split_env_flags("TILELANG_RISCV_LDFLAGS")
    return options


def is_cross_compiling(target: TargetLike) -> bool:
    """Return whether ``target`` needs a cross toolchain on this machine."""
    from tvm.target import Target

    if not isinstance(target, Target):
        try:
            target = Target(target)
        except Exception:
            return False
    if not target_is_riscv(target):
        return False
    import platform

    return platform.machine().lower() not in ("riscv64", "riscv64gc")


def create_riscv_shared(output: str, objects: Sequence[str], options: Sequence[str] | None = None, cc: str | None = None):
    """``fcompile`` compatible entry point that links RISC-V objects.

    Signature matches what :meth:`tvm.runtime.Module.export_library` expects
    (``fcompile(output, objects, **kwargs)``).  We assemble the command line
    ourselves instead of reusing ``tvm.contrib.cc.create_shared`` because that
    helper places ``options`` *before* the object files, which silently drops
    static archives (the linker discards archive members that satisfy no
    undefined symbol *yet*).  Library tokens (``-lfoo``, ``libfoo.a``,
    ``libfoo.so``) found in the options are therefore moved after the objects.

    Flag sources (see module docstring): ``TILELANG_RISCV_SYSROOT``,
    ``TILELANG_RISCV_CFLAGS``, ``TILELANG_RISCV_LDFLAGS``.  Library tokens in
    any of them are appended after the object files.
    """
    cc = cc or find_riscv_cc()
    if cc is None:
        raise RuntimeError(
            "No riscv64 cross compiler found. Install gcc-riscv64-linux-gnu or set TILELANG_RISCV_GCC to the compiler driver."
        )
    tokens = [*riscv_link_options(), *(options or ())]
    pre_flags: list[str] = []
    post_libs: list[str] = []
    for token in tokens:
        if token.startswith("-l") or token.endswith(".a") or token.endswith(".so"):
            post_libs.append(token)
        else:
            pre_flags.append(token)
    cmd = [cc, "-shared", "-fPIC", "-o", output, *pre_flags, *(str(obj) for obj in objects), *post_libs]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Linking RISC-V shared library failed: {' '.join(cmd)}\n{proc.stderr}")


create_riscv_shared.output_format = "so"


def riscv_export_kwargs(target: TargetLike | None = None) -> dict:
    """Export kwargs (``fcompile``) for producing a RISC-V shared library."""
    if target is not None and not is_cross_compiling(target):
        # Native RISC-V host: the default host-toolchain linking is correct.
        return {}
    return {"fcompile": create_riscv_shared}


def export_riscv_library(module, file_name: str, target: TargetLike | None = None, **kwargs) -> None:
    """Export a compiled TVM module tree as a RISC-V shared library.

    Parameters
    ----------
    module : tvm.runtime.Module
        The compiled module tree (``CompiledArtifact.rt_mod`` or the
        ``Executable`` produced by ``tilelang.compile``).
    file_name : str
        Output path, conventionally ``*.so``.
    target : optional
        The compilation target; used to decide whether the cross toolchain is
        required.  Defaults to cross linking (safe for export-only flows).
    """
    if hasattr(module, "mod") and hasattr(module, "export_library"):
        # tvm.runtime.Executable wrapper
        module = module.mod
    if hasattr(module, "export_library"):
        module.export_library(file_name, **riscv_export_kwargs(target), **kwargs)
    else:
        raise TypeError(f"Cannot export {type(module)} as a shared library")


def riscv_target_triple() -> str:
    """The LLVM triple used for all RISC-V codegen."""
    return RISCV64_TRIPLE


def verify_riscv_cc(cc: str | None = None) -> str:
    """Sanity-check a cross compiler; returns its ``-dumpmachine`` output."""
    cc = cc or find_riscv_cc()
    if cc is None:
        raise RuntimeError("No riscv64 cross compiler found")
    out = subprocess.run([cc, "-dumpmachine"], check=True, capture_output=True, text=True)
    machine = out.stdout.strip()
    if not machine.startswith("riscv64"):
        raise RuntimeError(f"Compiler {cc} targets {machine}, expected riscv64-*")
    return machine
