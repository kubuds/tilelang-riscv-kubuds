"""Shared fixtures and helpers for the RISC-V backend tests.

The RISC-V correctness tests work in three stages:

1. compile the TileLang kernel for a RISC-V target on the build host
   (``tilelang.compile`` with e.g. ``target="sg2042"``),
2. export the compiled module as a riscv64 shared library
   (:func:`tilelang.riscv.export.export_riscv_library`),
3. execute it under QEMU user-mode emulation
   (:func:`tilelang.riscv.qemu.run_kernel`) and compare against a numpy
   reference.

All external tooling (cross toolchain, QEMU, harness binary) is optional: when
it is missing the affected tests are skipped instead of failing, so the suite
stays runnable on plain x86 CI hosts.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

import tilelang
import tilelang.riscv
import tilelang.riscv.qemu as tlqemu
from tilelang.riscv.export import find_riscv_cc


def _cross_cc_available() -> bool:
    return find_riscv_cc() is not None


def _qemu_available() -> bool:
    try:
        tlqemu.find_qemu()
        return True
    except RuntimeError:
        return False


def _harness_available() -> bool:
    try:
        tlqemu.default_harness_path()
        return True
    except RuntimeError:
        return False


requires_cross_cc = pytest.mark.skipif(not _cross_cc_available(), reason="riscv64 cross compiler not available")
requires_qemu = pytest.mark.skipif(not _qemu_available(), reason="qemu-riscv64 not available")
requires_harness = pytest.mark.skipif(not _harness_available(), reason="riscv kernel harness not built")
requires_qemu_stack = pytest.mark.skipif(
    not (_cross_cc_available() and _qemu_available() and _harness_available()),
    reason="RISC-V QEMU execution stack (cross cc + qemu + harness) not available",
)


def compile_riscv_kernel(func, target="riscv64", out_idx=None, **kwargs):
    """Compile a TileLang kernel for a RISC-V target (cross or native)."""
    return tilelang.compile(func, target=target, out_idx=out_idx, execution_backend="tvm_ffi", **kwargs)


def export_kernel(kernel, path, target=None):
    """Export a compiled kernel as a riscv64 shared library."""
    executable = kernel.adapter.get_exportable_executable()
    target = target if target is not None else kernel.target
    tilelang.riscv.export.export_riscv_library(executable, str(path), target=target)
    return path


def run_qemu(kernel, args, lib_path, func_name="main", vlen=None, target=None):
    """Export ``kernel`` to ``lib_path`` and invoke it under QEMU.

    ``args`` follows :func:`tilelang.riscv.qemu.run_kernel`: one entry per
    kernel parameter, in declaration order.
    """
    export_kernel(kernel, lib_path, target=target)
    return tlqemu.run_kernel(lib_path, func_name, args, vlen=vlen)


def np_assert_close(actual: np.ndarray, expected: np.ndarray, rtol=1e-4, atol=1e-4):
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} vs {expected.shape}"
    assert actual.dtype == expected.dtype, f"dtype mismatch: {actual.dtype} vs {expected.dtype}"
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
