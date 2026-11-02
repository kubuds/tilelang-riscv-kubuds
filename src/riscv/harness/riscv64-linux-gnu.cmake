# CMake toolchain file for cross-compiling the TileLang RISC-V harness (and the
# bundled tvm-ffi runtime) to riscv64-linux-gnu.
#
# The compiler driver defaults to `riscv64-linux-gnu-gcc`/`-g++`; override via
# the TILELANG_RISCV_GCC / TILELANG_RISCV_GXX environment variables (or by
# setting CMAKE_C_COMPILER / CMAKE_CXX_COMPILER on the command line).
#
# TILELANG_RISCV_SYSROOT may point at a riscv64 sysroot; otherwise the cross
# toolchain's default sysroot layout is used.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(NOT DEFINED CMAKE_C_COMPILER)
  if(DEFINED ENV{TILELANG_RISCV_GCC})
    set(CMAKE_C_COMPILER "$ENV{TILELANG_RISCV_GCC}")
  else()
    find_program(_TILELANG_RISCV_GCC NAMES riscv64-linux-gnu-gcc riscv64-linux-gnu-gcc-14 riscv64-linux-gnu-gcc-13 riscv64-linux-gnu-gcc-12)
    if(_TILELANG_RISCV_GCC)
      set(CMAKE_C_COMPILER "${_TILELANG_RISCV_GCC}")
    else()
      set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)
    endif()
  endif()
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
  if(DEFINED ENV{TILELANG_RISCV_GXX})
    set(CMAKE_CXX_COMPILER "$ENV{TILELANG_RISCV_GXX}")
  else()
    get_filename_component(_tilelang_riscv_cxx_dir "${CMAKE_C_COMPILER}" DIRECTORY)
    get_filename_component(_tilelang_riscv_cxx_name "${CMAKE_C_COMPILER}" NAME)
    string(REPLACE "gcc" "g++" _tilelang_riscv_cxx_name "${_tilelang_riscv_cxx_name}")
    if(EXISTS "${_tilelang_riscv_cxx_dir}/${_tilelang_riscv_cxx_name}")
      set(CMAKE_CXX_COMPILER "${_tilelang_riscv_cxx_dir}/${_tilelang_riscv_cxx_name}")
    else()
      set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)
    endif()
  endif()
endif()

if(DEFINED ENV{TILELANG_RISCV_SYSROOT})
  set(CMAKE_SYSROOT "$ENV{TILELANG_RISCV_SYSROOT}")
  list(APPEND CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# RV64GC baseline: the harness itself must run on every RISC-V Linux box
# (including the emulation targets), so keep it to the "G" + "C" extensions.
set(CMAKE_C_FLAGS_INIT "-march=rv64gc -mabi=lp64d")
set(CMAKE_CXX_FLAGS_INIT "-march=rv64gc -mabi=lp64d")
