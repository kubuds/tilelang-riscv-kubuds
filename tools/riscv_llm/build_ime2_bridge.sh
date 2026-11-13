#!/bin/bash
# Build the IME2 Q4 bridge that the "ime2q4" blob layout calls.
#
# A single .so holds all three entry points (repack, M=1 gemv, M=4..16 gemm)
# sharing ONE 8-worker spert stream.  Two streams oversubscribe the 8-core A100
# cluster (output "!!!!", prefill 30 s); see tools/riscv_llm/ime2_all.cpp.
#
#   SPACEMIT_LLAMA=/work/.../spacemit-llama.cpp ./tools/riscv_llm/build_ime2_bridge.sh
#   -> build_riscv/libime2bridge.so
set -e

LL=${SPACEMIT_LLAMA:-/work/kubuds/hermes/spacemit-llama.cpp}
OUT=${OUT:-$PWD/build_riscv/libime2bridge.so}
HERE=$(cd "$(dirname "$0")" && pwd)
OBJ=$LL/build/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/spacemit

mkdir -p "$(dirname "$OUT")" /tmp/ime2_inc/ggml-spacemit
ln -sf "$LL/ggml/src/ggml-cpu/spacemit/repack.h" /tmp/ime2_inc/ggml-spacemit/repack.h

g++ -O3 -fPIC -shared -march=rv64gcv -mabi=lp64d -fopenmp \
  -o "$OUT" "$HERE/ime2_all.cpp" \
  "$OBJ/ime2_kernels.cpp.o" "$OBJ/rvv_kernels.cpp.o" "$OBJ/repack.cpp.o" \
  "$OBJ/ime1_kernels.cpp.o" \
  -I "$LL/ggml/include" -I "$LL/ggml/src" -I "$LL/ggml/src/ggml-cpu" \
  -I "$LL/ggml/src/ggml-cpu/spacemit" -I /tmp/ime2_inc \
  -Wl,--no-as-needed -L "$LL/build/bin" -lggml -lspert -lm

echo "wrote $OUT"
