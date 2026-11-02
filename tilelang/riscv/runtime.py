"""Runtime CPU-capability detection for RISC-V hosts.

Used to pick the best TileLang RISC-V preset on real silicon (SG2042,
SpacemiT K3 X100 / A100) and to dispatch between kernel variants at runtime.

Detection sources, in order:

1. ``riscv_hwprobe`` (Linux >= 6.5) -- the authoritative interface for
   ISA-extension queries,
2. ``/proc/cpuinfo`` -- the ``isa`` line advertises the enabled extensions,
3. ``os.uname``/platform -- architecture only, no vector information.

VLEN (the vector register length) is not directly exposed by ``/proc/cpuinfo``;
``riscv_hwprobe`` reports extension presence but not VLEN either, so
:func:`detect_vlen` returns the *minimum guaranteed* VLEN implied by the
reported ``zvl*`` extensions, or ``None`` when no vector extension is present.
Kernels that depend on the exact VLEN (e.g. tuned for X100's 256-bit or
A100's 1024-bit datapaths) should therefore still be selected explicitly via
the ``k3-x100``/``k3-a100`` presets or through a platform-level configuration.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re

# riscv_hwprobe syscall (kernel >= 6.5, arch-generic number on riscv64).
__NR_riscv_hwprobe = 258
RISCV_HWPROBE_KEY_IMA_EXT_0 = 1
RISCV_HWPROBE_HWCAP_V = 1 << 2  # 'V' bit of IMA_EXT_0 (as of Linux 6.5)


def is_riscv64_host() -> bool:
    return platform.machine().lower() in ("riscv64", "riscv64gc")


def _hwprobe_ima_ext0() -> int | None:
    """Query the IMA+extension bitmask via riscv_hwprobe; None when unsupported."""

    if not is_riscv64_host():
        return None
    # struct riscv_hwprobe { u64 key; u64 value; } pairs, zero-key terminated.
    Pair = ctypes.c_uint64 * 2
    pairs = (Pair * 1)((RISCV_HWPROBE_KEY_IMA_EXT_0, 0))
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        ret = libc.syscall(__NR_riscv_hwprobe, pairs, 1, 0, None, 0)
    except Exception:
        return None
    if ret != 0:
        return None
    return int(pairs[0][1])


def _proc_cpuinfo_isa() -> str | None:
    try:
        with open("/proc/cpuinfo") as file:
            for line in file:
                if line.startswith("isa"):
                    _, _, rest = line.partition(":")
                    return rest.strip()
    except OSError:
        return None
    return None


def detect_vector() -> bool:
    """Whether the host implements a usable RISC-V Vector extension."""

    value = _hwprobe_ima_ext0()
    if value is not None:
        return bool(value & RISCV_HWPROBE_HWCAP_V)
    isa = _proc_cpuinfo_isa()
    if isa:
        tokens = re.split(r"[_\s]", isa.lower())
        return "v" in tokens
    return False


def detect_vlen() -> int | None:
    """Best-effort architectural VLEN in bits (None when no V extension)."""

    if not detect_vector():
        return None
    isa = _proc_cpuinfo_isa()
    vlen = 128  # minimum guaranteed by the V extension
    if isa:
        for token in re.split(r"[_\s]", isa.lower()):
            match = re.fullmatch(r"zvl(\d+)b", token)
            if match:
                vlen = max(vlen, int(match.group(1)))
    return vlen


def detect_bitmanip() -> bool:
    """Whether the scalar Zba/Zbb/Zbc/Zbs bit-manipulation set is present."""

    value = _hwprobe_ima_ext0()
    if value is not None:
        # Zba=bit 3, Zbb=bit 4, Zbc=bit 5, Zbs=bit 6 of IMA_EXT_0.
        return (value & (0xF << 3)) == (0xF << 3)
    isa = _proc_cpuinfo_isa()
    if isa:
        tokens = set(re.split(r"[_\s]", isa.lower()))
        return {"zba", "zbb", "zbc", "zbs"} <= tokens
    return False


def detect_preset() -> str:
    """Suggest a TileLang RISC-V preset for the current host.

    Conservative: without vector support (or on the T-Head C920's incompatible
    RVV 0.7.1 draft) the RV64GC baseline is suggested.
    """

    if not is_riscv64_host():
        raise RuntimeError(f"not a RISC-V host: {platform.machine()}")
    if not detect_vector():
        return "riscv64"
    vlen = detect_vlen() or 128
    if vlen >= 1024:
        return "k3-a100"
    if vlen >= 256:
        return "k3-x100"
    return "riscv64"
