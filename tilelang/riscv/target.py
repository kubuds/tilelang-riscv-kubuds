"""RISC-V target detection and normalization.

This module teaches TileLang about RISC-V targets.  TVM itself already knows how
to generate RISC-V code through the ``llvm`` target kind (``-mtriple=riscv64...``),
so all this module has to do is:

* provide friendly preset names for the silicon we care about
  (``riscv64``/``sg2042`` for the RV64GC baseline, ``k3-x100`` for RVV 1.0 with
  ``VLEN=256`` and ``k3-a100`` for RVV 1.0 with ``VLEN=1024`` plus the
  SpacemiT IME-2 matrix engine), and
* auto-detect a sensible default when TileLang runs natively on a RISC-V host.

Every preset is lowered to a plain ``llvm`` target with an explicit
``-mtriple``/``-mcpu``/``-mattr`` set, so the rest of the compiler stack needs
no knowledge of the preset names.  Keep the ISA attribute strings here in sync
with the hardware documentation:

* SG2042: 64x T-Head C920 cores, RV64GC (the C920 additionally implements the
  draft RVV 0.7.1 vector extension, which is *not* compatible with the LLVM
  ``+v`` (RVV 1.0) codegen -- hence the baseline preset deliberately keeps the
  vector extension disabled).
* K3 X100: RVV 1.0, VLEN=256, ELEN=64, RVA23 (see the SpacemiT K3 paper,
  "SpacemiT K3: A RVA23 RISC-V AI CPU with 60 TOPS AI Compute").
* K3 A100: RVV 1.0, VLEN=1024, plus the SpacemiT-IME 2.0 matrix engine shared
  by two cores.  The IME matrix-multiply intrinsics are expressed in TIR (see
  :mod:`tilelang.riscv.ime`) and lowered against the ``libime2`` runtime ABI.

References
----------
The ISA extension strings follow the RISC-V Unprivileged ISA specification and
the LLVM RISC-V backend attribute documentation.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer

RISCV64_TRIPLE = "riscv64-unknown-linux-gnu"

# Baseline scalar extensions ("GC") plus the CSR/fencei extensions that the
# "G" shorthand expands to on modern toolchains.
_RV64GC_ATTRS = ("+m", "+a", "+f", "+d", "+c", "+zicsr", "+zifencei")

# RVA23 makes the scalar bit-manipulation extensions mandatory.  Both K3 core
# clusters (X100 and A100) are RVA23 conformant, so the K3 presets enable them.
_RVA23_BITMANIP_ATTRS = ("+zba", "+zbb", "+zbc", "+zbs")


@dataclass(frozen=True, slots=True)
class RISCVTargetSpec:
    """One named RISC-V SoC/CPU preset."""

    name: str
    description: str
    mcpu: str = "generic-rv64"
    mattr: tuple[str, ...] = field(default=())
    # Vector length in bits advertised by the hardware (0 for scalar targets).
    vlen: int = 0
    # SpacemiT IME matrix engine generation (0, 1 or 2).
    ime: int = 0

    def to_target(self, *, extra_mattr: tuple[str, ...] = (), opt_level: int | None = None) -> Target:
        attrs: dict[str, Any] = {
            "kind": "llvm",
            "mtriple": RISCV64_TRIPLE,
            "mcpu": self.mcpu,
        }
        mattr = list(dict.fromkeys([*self.mattr, *extra_mattr]))
        if mattr:
            attrs["mattr"] = mattr
        if opt_level is not None:
            attrs["opt-level"] = opt_level
        return Target(attrs)


# The canonical preset table.  ``riscv64`` is the conservative baseline that
# runs on every RV64GC machine; the named SoC presets add the optional
# extensions the respective silicon provides.
RISCV_TARGET_SPECS: dict[str, RISCVTargetSpec] = {
    "riscv64": RISCVTargetSpec(
        name="riscv64",
        description="Generic RV64GC Linux target (scalar baseline)",
        mattr=_RV64GC_ATTRS,
    ),
    "sg2042": RISCVTargetSpec(
        name="sg2042",
        description="Sophgo SG2042 (64x T-Head C920), RV64GC scalar baseline",
        # The C920's vector unit implements the incompatible RVV 0.7.1 draft,
        # so no "+v" here.  LLVM has no dedicated c920 CPU model; generic-rv64
        # with explicit GC extensions is the safe baseline.
        mattr=_RV64GC_ATTRS,
    ),
    "k3-x100": RISCVTargetSpec(
        name="k3-x100",
        description="SpacemiT K3 X100 core cluster, RVV 1.0 VLEN=256",
        mattr=(*_RV64GC_ATTRS, *_RVA23_BITMANIP_ATTRS, "+v", "+zvl256b"),
        vlen=256,
    ),
    "k3-a100": RISCVTargetSpec(
        name="k3-a100",
        description="SpacemiT K3 A100 AI-CPU cluster, RVV 1.0 VLEN=1024 + IME-2.0",
        mattr=(*_RV64GC_ATTRS, *_RVA23_BITMANIP_ATTRS, "+v", "+zvl1024b"),
        vlen=1024,
        ime=2,
    ),
}

_ALIASES = {
    "riscv": "riscv64",
    "riscv64-gc": "riscv64",
    "rv64gc": "riscv64",
    "spacemit-k3-x100": "k3-x100",
    "spacemit-k3-a100": "k3-a100",
    "k3_x100": "k3-x100",
    "k3_a100": "k3-a100",
}


def _resolve_preset(name: str) -> RISCVTargetSpec | None:
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    return RISCV_TARGET_SPECS.get(key)


def target_attr(target: Target, name: str, default=None):
    """Read a target attribute across TVM attr container flavours."""

    attrs = target.attrs
    if hasattr(attrs, "get"):
        value = attrs.get(name, None)
        return default if value is None else value
    return getattr(attrs, name, default)


def is_riscv64_triple(triple: str | None) -> bool:
    """Return whether an LLVM triple targets 64-bit RISC-V Linux/ELF."""
    if not triple:
        return False
    triple = triple.strip().lower()
    return triple.startswith("riscv64")


def target_is_riscv(target: Target) -> bool:
    """Return whether a TVM target will generate RISC-V code."""
    if target.kind.name != "llvm":
        return False
    return is_riscv64_triple(target_attr(target, "mtriple"))


def lookup_preset(target: TargetLike) -> RISCVTargetSpec | None:
    """Best-effort lookup of the preset a target was created from.

    Matching is done on the encoded ``mcpu``/``mattr``/``mtriple`` combination,
    so it keeps working for targets that were serialized and re-parsed.
    """
    if not isinstance(target, Target):
        try:
            target = Target(target)
        except Exception:
            return None
    if not target_is_riscv(target):
        return None
    mcpu = str(target_attr(target, "mcpu", "") or "")
    mattr = tuple(str(x) for x in (target_attr(target, "mattr", ()) or ()))
    candidates = [spec for spec in RISCV_TARGET_SPECS.values() if spec.mcpu == mcpu and set(spec.mattr) == set(mattr)]
    return candidates[0] if len(candidates) == 1 else None


def vlen_of(target: Target) -> int | None:
    """Return the minimum vector register length (bits) a target requires.

    Derived from the ``+zvl<bits>b`` attributes, ``None`` when the target does
    not enable the V extension.
    """
    if not isinstance(target, Target):
        target = Target(target)
    mattr = [str(x).lower() for x in (target_attr(target, "mattr", ()) or ())]
    if not any(x.lstrip("+-") == "v" for x in mattr):
        return None
    vlen = 128
    for item in mattr:
        name = item.lstrip("+-")
        if name.startswith("zvl") and name.endswith("b") and name[3:-1].isdigit():
            vlen = max(vlen, int(name[3:-1]))
    return vlen


def _mtriple_of(target_like: TargetLike) -> str | None:
    if isinstance(target_like, Target):
        return target_attr(target_like, "mtriple")
    if isinstance(target_like, dict):
        return target_like.get("mtriple")
    if isinstance(target_like, str):
        for token in target_like.split():
            if token.startswith("mtriple="):
                return token.split("=", 1)[1]
        return None
    return None


def riscv_target(
    soc: str = "riscv64",
    *,
    mcpu: str | None = None,
    mattr: list[str] | None = None,
    opt_level: int | None = None,
) -> Target:
    """Build a RISC-V target for ``soc`` (see :data:`RISCV_TARGET_SPECS`)."""
    spec = _resolve_preset(soc)
    if spec is None:
        known = ", ".join(sorted(RISCV_TARGET_SPECS))
        raise ValueError(f"Unknown RISC-V target preset {soc!r}. Known presets: {known}")
    target = spec.to_target(extra_mattr=tuple(mattr or ()), opt_level=opt_level)
    if mcpu is not None:
        export = dict(target.export())
        export["mcpu"] = mcpu
        export["keys"] = tuple(export.get("keys", ()))
        target = Target(export)
    return target


def normalize_riscv_target(target: TargetLike) -> Target | None:
    """Normalize the friendly RISC-V spellings to plain ``llvm`` targets.

    Accepted forms:

    * preset strings, e.g. ``"k3-x100"`` or ``"riscv64"``,
    * config dicts with ``kind="riscv64"`` and optional ``soc``/``mcpu``/
      ``mattr``/``opt-level`` overrides,
    * config dicts / targets that already are ``llvm`` with a RISC-V triple
      (returned unchanged so the normalizer stays idempotent).
    """
    if isinstance(target, str):
        spec = _resolve_preset(target)
        return spec.to_target() if spec is not None else None

    if isinstance(target, dict):
        kind = str(target.get("kind", "")).split()[0].lower()
        if kind in ("riscv", "riscv64"):
            spec = _resolve_preset(str(target.get("soc", "riscv64")))
            if spec is None:
                return None
            extra_mattr = tuple(str(x) for x in target.get("mattr", ()) or ())
            opt_level = target.get("opt-level")
            opt_level = int(opt_level) if opt_level is not None else None
            return spec.to_target(extra_mattr=extra_mattr, opt_level=opt_level)
        return None

    return None


def _detect_riscv_target() -> Target | None:
    """Auto-detection for natively running RISC-V hosts.

    We stay deliberately conservative and return the scalar RV64GC baseline:
    probing vector capabilities from user space is unreliable (e.g. the T-Head
    C920 in the SG2042 implements the incompatible RVV 0.7.1 draft) and mis-
    generating vector code would crash the process.  Pick ``k3-x100``/
    ``k3-a100`` explicitly on vector-capable hardware.
    """
    if platform.machine().lower() not in ("riscv64", "riscv64gc"):
        return None
    return _resolve_preset("riscv64").to_target()


register_target_detector("riscv", _detect_riscv_target, override=True)
register_target_normalizer("riscv", normalize_riscv_target, override=True)
