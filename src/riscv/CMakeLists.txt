# RISC-V backend: source files that are always safe to compile.
#
# Target attribute helpers have no optional system dependencies and are used
# by shared compiler passes (e.g. the loop vectorizer reads the RVV VLEN), so
# they are built unconditionally.

file(GLOB TILE_LANG_RISCV_SRCS
  src/riscv/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_RISCV_SRCS})
