"""Unit tests for RISC-V target normalization (no RISC-V tooling required)."""

import pytest

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.backend.module import create_backend_context
from tilelang.backend.target import determine_target
from tilelang.riscv.target import (
    RISCV64_TRIPLE,
    is_riscv64_triple,
    lookup_preset,
    riscv_target,
    target_attr,
    target_is_riscv,
    vlen_of,
)


def _mtriple(target) -> str:
    return str(target_attr(target, "mtriple", ""))


def _mattr(target) -> list[str]:
    return [str(x) for x in (target_attr(target, "mattr", ()) or ())]


def test_preset_strings_normalize_to_llvm_targets():
    for name, vlen in (("riscv64", None), ("sg2042", None), ("k3-x100", 256), ("k3-a100", 1024)):
        target = determine_target(name, return_object=True)
        assert target.kind.name == "llvm"
        assert _mtriple(target) == RISCV64_TRIPLE
        assert target_is_riscv(target)
        assert vlen_of(target) == vlen


def test_preset_aliases_resolve():
    assert _mtriple(determine_target("rv64gc", return_object=True)) == RISCV64_TRIPLE
    assert vlen_of(determine_target("spacemit-k3-x100", return_object=True)) == 256


def test_dict_config_form():
    target = determine_target({"kind": "riscv64", "soc": "k3-a100"}, return_object=True)
    assert vlen_of(target) == 1024
    # extra ISA attributes can be layered on top of the preset
    target = determine_target({"kind": "riscv64", "soc": "k3-x100", "mattr": ["+zvfh"]}, return_object=True)
    assert "+zvfh" in _mattr(target)


def test_unknown_preset_raises():
    with pytest.raises(Exception):
        riscv_target("sg2044")


def test_lookup_preset_roundtrip():
    target = riscv_target("k3-x100")
    assert lookup_preset(target) is not None
    assert lookup_preset(target).name == "k3-x100"


def test_is_riscv64_triple():
    assert is_riscv64_triple("riscv64-unknown-linux-gnu")
    assert is_riscv64_triple("riscv64-linux-gnu")
    assert not is_riscv64_triple("x86_64-linux-gnu")
    assert not is_riscv64_triple(None)


def test_llvm_dict_with_riscv_triple_is_detected():
    # Note: this TVM requires the JSON dict form for targets with attributes.
    target = tvm.target.Target({"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu", "mattr": ["+v", "+zvl256b"]})
    assert target_is_riscv(target)
    assert vlen_of(target) == 256


def test_backend_context_routes_riscv_to_riscv_backend():
    ctx = create_backend_context("k3-x100", execution_backend="tvm_ffi")
    assert ctx.name == "riscv"
    assert target_is_riscv(ctx.target)
    # host glue must target the same architecture as the kernel
    assert target_is_riscv(ctx.target_host)


def test_backend_context_keeps_native_llvm_on_cpu_backend():
    import platform

    ctx = create_backend_context("llvm", execution_backend="tvm_ffi")
    if platform.machine().lower() in ("riscv64", "riscv64gc"):
        # Native RISC-V host: the riscv backend serves the bare llvm target.
        assert ctx.name == "riscv"
    else:
        assert ctx.name == "cpu"


def test_vlen_of_scalar_target_is_none():
    assert vlen_of(determine_target("sg2042", return_object=True)) is None


if __name__ == "__main__":
    tilelang.testing.main()
