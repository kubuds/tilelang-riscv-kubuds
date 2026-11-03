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

# Step 3: SpacemiT IME2 (smt.vmadot) int8 GEMM.
#
# Built as a standalone shared library, not linked into libtilelang.so: the
# kernel needs -march=..._xsmtvdotii and must only ever run on the A100 cluster
# (cpu8-15, VLEN=1024).  Keeping it separate also means a build on a toolchain
# without the vendor extension still produces a working TileLang; only
# target=riscv,ime2=true kernels then fail to link.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64" AND NOT TARGET tilelang_riscv_ime2)
  set(TL_IME2_SRC  ${CMAKE_CURRENT_SOURCE_DIR}/src/tl_templates/riscv/ime2/ime2_gemm_k3.c)
  set(TL_IME2_LIB  ${CMAKE_BINARY_DIR}/lib/libtl_riscv_ime2.so)
  set(TL_IME2_ARCH "rv64gcv_zfh_zvfh_zba_zbb_zbs_zicbop_zihintpause_zawrs_zicond_xsmtvdotii")
  add_custom_command(
    OUTPUT ${TL_IME2_LIB}
    COMMAND ${CMAKE_C_COMPILER} -O3 -fPIC -shared -march=${TL_IME2_ARCH}
            -mabi=lp64d -fopenmp ${TL_IME2_SRC} -o ${TL_IME2_LIB}
    DEPENDS ${TL_IME2_SRC}
    COMMENT "Building libtl_riscv_ime2.so (A100 vmadot kernels)"
    VERBATIM)
  add_custom_target(tilelang_riscv_ime2 ALL DEPENDS ${TL_IME2_LIB})
endif()
