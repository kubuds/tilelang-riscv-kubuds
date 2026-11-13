#!/bin/bash
# Rebuild libtilelang.so from scratch (build_riscv/ is NOT persistent -- it has
# vanished overnight before; see the tilelang-riscv-build skill).
#
# The tricky part is Python: cmake's generic find_package(Python) resolved to
# /usr/bin/python3 (3.14), which has no Cython and whose ABI does not match the
# venv.  The venv python (3.12.13, uv-managed) bundles its own headers and
# libpython, so point every Python variable at it explicitly.

USE_ALTERNATIVE_LINKER=OFF matters: once lld is installed (it is, for IREE),
TVM's AUTO picks it and the gcc-compiled objects fail to link (ld.lld "relocation
refers to a symbol in a discarded section").  USE_ASCEND=OFF skips the NPU stubs.
set -e
cd "$(dirname "$0")/../.."          # repo root

UV=/home/kubuds/.local/share/uv/python/cpython-3.12.13-linux-riscv64gc-gnu
VPY=/work/kubuds/hermes/flagos_ws/.venv/bin/python

mkdir -p build_riscv && cd build_riscv
cmake .. \
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_ASCEND=OFF \
  -DUSE_LLVM=OFF -DUSE_ALTERNATIVE_LINKER=OFF \
  -DTILELANG_USE_CUDA_STUBS=OFF \
  -DPython_EXECUTABLE="$VPY" \
  -DPython_INCLUDE_DIR="$UV/include/python3.12" \
  -DPython_LIBRARY="$UV/lib/libpython3.12.so" \
  -DPython3_EXECUTABLE="$VPY" \
  -DPython3_INCLUDE_DIR="$UV/include/python3.12" \
  -DPython3_LIBRARY="$UV/lib/libpython3.12.so" \
  > /tmp/rebuild_cmake.log 2>&1
echo "cmake done: $(tail -1 /tmp/rebuild_cmake.log)"

make -j8 > /tmp/rebuild_make.log 2>&1
echo "make exit: $?"

cd .. && ln -sfn build_riscv build
ls -l lib/libtilelang.so
