"""Unit tests for RISC-V runtime capability detection (hardware mocked)."""

import pytest

import tilelang
import tilelang.testing
from tilelang.riscv import runtime


def test_detect_vector_from_hwprobe(monkeypatch):
    monkeypatch.setattr(runtime, "_hwprobe_ima_ext0", lambda: runtime.RISCV_HWPROBE_HWCAP_V)
    monkeypatch.setattr(runtime, "_proc_cpuinfo_isa", lambda: None)
    assert runtime.detect_vector()
    assert runtime.detect_vlen() == 128  # minimum guaranteed by V


def test_no_vector_from_hwprobe(monkeypatch):
    monkeypatch.setattr(runtime, "_hwprobe_ima_ext0", lambda: 0)
    monkeypatch.setattr(runtime, "_proc_cpuinfo_isa", lambda: None)
    assert not runtime.detect_vector()
    assert runtime.detect_vlen() is None


def test_detect_vlen_from_cpuinfo_zvl(monkeypatch):
    monkeypatch.setattr(runtime, "_hwprobe_ima_ext0", lambda: None)
    monkeypatch.setattr(runtime, "_proc_cpuinfo_isa", lambda: "rv64gcv_zvl256b_zba_zbb_zbc_zbs")
    assert runtime.detect_vector()
    assert runtime.detect_vlen() == 256
    assert runtime.detect_bitmanip()


def test_detect_vlen_zvl1024b(monkeypatch):
    monkeypatch.setattr(runtime, "_hwprobe_ima_ext0", lambda: None)
    monkeypatch.setattr(runtime, "_proc_cpuinfo_isa", lambda: "rv64gcv_zvl1024b")
    assert runtime.detect_vlen() == 1024


def test_c920_style_isa_without_v(monkeypatch):
    """The C920's RVV 0.7.1 draft may advertise 'v'; the preset stays scalar-safe."""
    monkeypatch.setattr(runtime, "is_riscv64_host", lambda: True)
    monkeypatch.setattr(runtime, "_hwprobe_ima_ext0", lambda: None)
    monkeypatch.setattr(runtime, "_proc_cpuinfo_isa", lambda: "rv64imafdc")
    assert not runtime.detect_vector()
    assert runtime.detect_preset() == "riscv64"


def test_detect_preset_mapping(monkeypatch):
    monkeypatch.setattr(runtime, "is_riscv64_host", lambda: True)
    cases = [(None, "riscv64"), (256, "k3-x100"), (1024, "k3-a100")]
    for vlen, expected in cases:
        monkeypatch.setattr(runtime, "_hwprobe_ima_ext0", lambda: None)
        monkeypatch.setattr(runtime, "detect_vector", lambda vlen=vlen: vlen is not None)
        monkeypatch.setattr(runtime, "detect_vlen", lambda vlen=vlen: vlen)
        assert runtime.detect_preset() == expected


def test_detect_preset_requires_riscv_host(monkeypatch):
    monkeypatch.setattr(runtime, "is_riscv64_host", lambda: False)
    with pytest.raises(RuntimeError):
        runtime.detect_preset()


if __name__ == "__main__":
    tilelang.testing.main()
