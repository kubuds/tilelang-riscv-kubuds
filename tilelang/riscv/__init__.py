"""TileLang RISC-V backend.

Public surface:

* :mod:`tilelang.riscv.target` -- target presets (``riscv64``, ``sg2042``,
  ``k3-x100``, ``k3-a100``), auto-detection and normalization,
* :mod:`tilelang.riscv.export` -- cross-linking compiled kernels into RISC-V
  shared libraries,
* :mod:`tilelang.riscv.backend` -- the backend manifest registration.

Importing this package registers the backend with the TileLang compiler.
"""

from . import target as target  # noqa: F401  (registers detector/normalizer)
from . import backend as backend  # noqa: F401  (registers the backend)
from .target import (  # noqa: F401
    RISCV64_TRIPLE,
    RISCV_TARGET_SPECS,
    RISCVTargetSpec,
    is_riscv64_triple,
    lookup_preset,
    riscv_target,
    target_is_riscv,
    vlen_of,
)
from .export import (  # noqa: F401
    export_riscv_library,
    find_riscv_cc,
    is_cross_compiling,
    riscv_export_kwargs,
    verify_riscv_cc,
)
