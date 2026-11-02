"""QEMU user-mode test runner for RISC-V kernels.

Cross-compiled TileLang kernels cannot be executed in the build process (they
target riscv64 while the test process runs on the build host).  For correctness
testing we run the :mod:`tilelang.riscv.export`-produced kernel libraries under
``qemu-riscv64`` (QEMU user-mode emulation) inside a small C++ harness that
speaks the TVM FFI packed-function ABI -- the very same ABI used on real
silicon.

The harness binary is built from ``src/riscv/harness`` (see
``src/riscv/harness/CMakeLists.txt``).  Tooling paths can be pinned through the
environment:

* ``TILELANG_QEMU`` -- qemu-riscv64 binary (default: search PATH),
* ``TILELANG_RISCV_HARNESS`` -- prebuilt harness binary,
* ``TILELANG_RISCV_SYSROOT`` -- riscv64 sysroot passed to QEMU as ``-L``.

This module is test/tooling support only; production execution goes through the
regular TVM FFI adapter on the target machine.
"""

from __future__ import annotations

import glob
import os
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_MAGIC = b"TLR1"

# dlpack type codes (dlpack.h: DLDataTypeCode)
_DL_INT = 0
_DL_UINT = 1
_DL_FLOAT = 2
_DL_BFLOAT = 4
_DL_BOOL = 6


def find_qemu(cpu_hint: str | None = None) -> str:
    explicit = os.environ.get("TILELANG_QEMU")
    if explicit:
        return explicit
    for name in ("qemu-riscv64", "qemu-riscv64-static"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("qemu-riscv64 not found; install qemu-user(-static) or set TILELANG_QEMU")


def default_harness_path() -> Path:
    env = os.environ.get("TILELANG_RISCV_HARNESS")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "build" / "riscv-harness" / "riscv_kernel_harness",
        here.parents[2] / "build" / "riscv-harness" / "riscv_kernel_harness",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise RuntimeError(
        "riscv_kernel_harness not found; build it with "
        "`cmake -S src/riscv/harness -B build/riscv-harness "
        "-DCMAKE_TOOLCHAIN_FILE=src/riscv/harness/riscv64-linux-gnu.cmake && "
        "cmake --build build/riscv-harness` or set TILELANG_RISCV_HARNESS"
    )


def qemu_prefix() -> str | None:
    """Directory layout expected by QEMU ``-L`` (contains ``lib/ld-linux-...``).

    ``TILELANG_RISCV_SYSROOT`` may be given either as the QEMU prefix itself
    or as the toolchain sysroot root (containing ``usr/riscv64-linux-gnu``);
    both spellings are common, so probe both layouts.
    """

    sysroot = os.environ.get("TILELANG_RISCV_SYSROOT")
    if not sysroot:
        return None
    candidates = [
        sysroot,
        os.path.join(sysroot, "usr", "riscv64-linux-gnu"),
        os.path.join(sysroot, "riscv64-linux-gnu"),
    ]
    for cand in candidates:
        if glob.glob(os.path.join(cand, "lib", "ld-linux-riscv64*")):
            return cand
    return sysroot


@dataclass
class QEMUTensor:
    """One tensor argument of the kernel call.

    ``data`` is both the input payload (when ``is_input``) and the buffer that
    receives results (when ``not is_input``).  Pass a preallocated array of the
    right dtype/shape for outputs.
    """

    data: np.ndarray
    is_input: bool = True
    name: str = ""
    meta: dict = field(default_factory=dict)


def _dl_dtype_of(arr: np.ndarray) -> tuple[int, int, int]:
    dtype = arr.dtype
    if dtype == np.bool_:
        return (_DL_BOOL, 8, 1)
    if dtype == np.dtype("uint8"):
        return (_DL_UINT, 8, 1)
    if dtype == np.dtype("int8"):
        return (_DL_INT, 8, 1)
    if dtype == np.dtype("uint16"):
        return (_DL_UINT, 16, 1)
    if dtype == np.dtype("int16"):
        return (_DL_INT, 16, 1)
    if dtype == np.dtype("uint32"):
        return (_DL_UINT, 32, 1)
    if dtype == np.dtype("int32"):
        return (_DL_INT, 32, 1)
    if dtype == np.dtype("uint64"):
        return (_DL_UINT, 64, 1)
    if dtype == np.dtype("int64"):
        return (_DL_INT, 64, 1)
    if dtype == np.dtype("float16"):
        return (_DL_FLOAT, 16, 1)
    if dtype == np.dtype("float32"):
        return (_DL_FLOAT, 32, 1)
    if dtype == np.dtype("float64"):
        return (_DL_FLOAT, 64, 1)
    if dtype == np.dtype("bfloat16") or str(dtype) == "bfloat16":
        return (_DL_BFLOAT, 16, 1)
    raise TypeError(f"unsupported numpy dtype for QEMU kernel call: {dtype}")


def _encode_request(args: list[QEMUTensor | int | float]) -> bytes:
    out = bytearray()
    out += _MAGIC
    out += struct.pack("<i", len(args))
    for arg in args:
        if isinstance(arg, bool):
            raise TypeError("bool is not a valid scalar argument; use int")
        if isinstance(arg, int):
            out += struct.pack("<i", 0)
            out += struct.pack("<q", arg)
            continue
        if isinstance(arg, float):
            out += struct.pack("<i", 2)
            out += struct.pack("<d", arg)
            continue
        code, bits, lanes = _dl_dtype_of(arg.data)
        out += struct.pack("<i", 1)
        out += struct.pack("<iiiii", 1 if arg.is_input else 0, code, bits, lanes, arg.data.ndim)
        for dim in arg.data.shape:
            out += struct.pack("<q", dim)
    for arg in args:
        if isinstance(arg, QEMUTensor) and arg.is_input:
            out += np.ascontiguousarray(arg.data).tobytes()
    return bytes(out)


def _decode_response(blob: bytes, args: list[QEMUTensor | int | float]) -> None:
    status = struct.unpack_from("<i", blob, 0)[0]
    if status != 0:
        raise RuntimeError(f"kernel execution failed under QEMU (status={status})")
    offset = 4
    for arg in args:
        if isinstance(arg, QEMUTensor) and not arg.is_input:
            nbytes = arg.data.nbytes
            chunk = blob[offset : offset + nbytes]
            if len(chunk) != nbytes:
                raise RuntimeError("truncated output payload from QEMU harness")
            arr = np.frombuffer(chunk, dtype=arg.data.dtype).reshape(arg.data.shape)
            arg.data[...] = arr
            offset += nbytes


def run_kernel(
    kernel_lib: str | os.PathLike,
    func_name: str,
    args: list[QEMUTensor | int | float],
    *,
    vlen: int | None = None,
    harness: str | os.PathLike | None = None,
    timeout: float = 300.0,
    extra_env: dict[str, str] | None = None,
) -> list[QEMUTensor]:
    """Run one kernel call under QEMU and return the output tensors.

    Parameters
    ----------
    kernel_lib:
        The exported riscv64 shared library (``.so``) produced by
        :func:`tilelang.riscv.export.export_riscv_library`.
    func_name:
        The packed-function entry to call (usually the kernel's global symbol).
    args:
        Kernel arguments in declaration order: :class:`QEMUTensor` for buffers,
        plain ``int`` for integer (shape) parameters and plain ``float`` for
        floating-point scalar parameters.  Output buffers are the entries with
        ``is_input=False`` and are filled in-place.
    vlen:
        Request a specific RVV vector length for the emulated CPU (``None``
        for the scalar RV64GC baseline).
    """
    harness_path = Path(harness) if harness is not None else default_harness_path()
    qemu = find_qemu()

    cmd = [qemu]
    if vlen is not None:
        cmd += ["-cpu", f"rv64,v=true,vlen={vlen}"]
    sysroot = qemu_prefix()
    if sysroot:
        cmd += ["-L", sysroot]
    cmd += [str(harness_path), str(kernel_lib), func_name]

    env = dict(os.environ)
    harness_dir = Path(harness_path).resolve().parent
    lib_dirs = [str(harness_dir), str(harness_dir / "lib")]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    if extra_env:
        env.update(extra_env)

    request = _encode_request(args)
    proc = subprocess.run(cmd, input=request, capture_output=True, timeout=timeout, env=env)
    if proc.stderr:
        # Surface harness diagnostics (e.g. tvm-ffi errors) without failing here.
        import sys

        print(proc.stderr.decode("utf-8", errors="replace"), file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"QEMU harness exited with status {proc.returncode}\n"
            f"cmd: {' '.join(cmd)}\nstderr:\n{proc.stderr.decode('utf-8', errors='replace')}"
        )
    _decode_response(proc.stdout, args)
    return [arg for arg in args if isinstance(arg, QEMUTensor) and not arg.is_input]
