"""RISC-V backend package.

Importing this module registers the ``riscv`` target kind backend with the
TileLang backend registry.  The heavy lifting (TIR -> C) is delegated to the
existing C host codegen; this package only owns the *ISA-specific* parts:
toolchain flags derived from target attributes, and (in later steps) RVV and
IME2 lowering.
"""

from __future__ import annotations

from .backend import BACKEND

__all__ = ["BACKEND"]
