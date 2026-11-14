#!/bin/bash
# Rebuild libtilelang.so from scratch (build_riscv/ is NOT persistent -- it has
# vanished overnight before; see the tilelang-riscv-build skill).
#
# The tricky part is Python: cmake's generic find_package(Python) resolved to
# /usr/bin/python3 (3.14), which has no Cython and whose ABI does not match the
# venv.  Point every Python variable at the interpreter that will import
# tilelang: $VPY if set, else the active virtualenv, else python3.  Interpreters
# built with uv bundle their own headers/libpython, so the include/library
# paths are derived from the interpreter itself rather than hardcoded.
#
# USE_ALTERNATIVE_LINKER=OFF matters: once lld is installed (it is, for IREE),
# TVM's AUTO picks it and the gcc-compiled objects fail to link (ld.lld
# "relocation refers to a symbol in a discarded section").  USE_ASCEND=OFF
# skips the NPU stubs.  USE_LLVM=OFF is required for the riscv target: with
# LLVM enabled the host glue is compiled to generic rv64i soft-float objects
# that cannot link with the -mabi=lp64d kernels (see tilelang/riscv/toolchain.py).
set -e
cd "$(dirname "$0")/../.."          # repo root

if [ -n "${VPY:-}" ]; then
  : # respect caller
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  VPY="$VIRTUAL_ENV/bin/python"
else
  VPY="$(command -v python3)"
fi
PY_INC=$("$VPY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
PY_LIBDIR=$("$VPY" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
PY_LDVER=$("$VPY" -c 'import sysconfig; print(sysconfig.get_config_var("LDVERSION"))')
PY_LIB=$(ls "$PY_LIBDIR"/libpython"$PY_LDVER".so* 2>/dev/null | head -1)
echo "python: $VPY"
echo "include: $PY_INC"
echo "library: ${PY_LIB:-<not found; FindPython will search>}"

mkdir -p build_riscv && cd build_riscv
CMAKE_ARGS=(
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_ASCEND=OFF
  -DUSE_LLVM=OFF -DUSE_ALTERNATIVE_LINKER=OFF
  -DTILELANG_USE_CUDA_STUBS=OFF
  -DPython_EXECUTABLE="$VPY" -DPython3_EXECUTABLE="$VPY"
)
if [ -n "$PY_INC" ]; then
  CMAKE_ARGS+=(-DPython_INCLUDE_DIR="$PY_INC" -DPython3_INCLUDE_DIR="$PY_INC")
fi
if [ -n "$PY_LIB" ]; then
  CMAKE_ARGS+=(-DPython_LIBRARY="$PY_LIB" -DPython3_LIBRARY="$PY_LIB")
fi
cmake .. "${CMAKE_ARGS[@]}" > /tmp/rebuild_cmake.log 2>&1
echo "cmake done: $(tail -1 /tmp/rebuild_cmake.log)"

make -j"$(nproc)" > /tmp/rebuild_make.log 2>&1
echo "make exit: $?"

cd .. && ln -sfn build_riscv build
ls -l build/lib/libtilelang.so
