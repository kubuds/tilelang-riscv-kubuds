# RISC-V backend: target kind registration and ISA-specific lowering.
#
# Mirrors src/cpu/CMakeLists.txt: this backend owns its own source list, and
# the top-level CMake only delegates to it.  `codegen/*.cc` is globbed so that
# step 2 (RVV) and step 3 (IME2) can add files without touching this file.

file(GLOB TILE_LANG_RISCV_SRCS
  src/riscv/target.cc
  src/riscv/codegen/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_RISCV_SRCS})
