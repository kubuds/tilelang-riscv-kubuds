#!/bin/bash
# Rebuild libtilelang.so after build_riscv/ vanished from the tree.
set -e
cd /work/kubuds/hermes/tilelang_riscv
mkdir -p build_riscv && cd build_riscv
cmake .. -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_LLVM=OFF \
  -DTILELANG_USE_CUDA_STUBS=OFF \
  -DPython3_EXECUTABLE=/work/kubuds/hermes/flagos_ws/.venv/bin/python3 \
  > /tmp/rebuild_cmake.log 2>&1
echo "cmake done: $(tail -1 /tmp/rebuild_cmake.log)"
make -j8 > /tmp/rebuild_make.log 2>&1
echo "make exit: $?"
ls -l lib/libtilelang.so 2>/dev/null
# restore the `build` symlink the dev loader expects
cd /work/kubuds/hermes/tilelang_riscv && ln -sfn build_riscv build
ls -l lib/libtilelang.so 2>/dev/null
