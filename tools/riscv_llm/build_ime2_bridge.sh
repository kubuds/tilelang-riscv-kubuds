#!/bin/bash
# Build the IME2 Q4 bridge that the "ime2q4" blob layout calls.
#
# Contains two entry points:
#   ime2_gemv_q4    -- M=1 decode GEMV (from the kubuds-riscv-mahiru branch,
#                      Apache-2.0; vendor spert kernels, MIT)
#   ime2_gemm_q4_m  -- M=4/M=8 batched GEMM for prefill (this repo)
#
# Both need the vendor spert kernel objects already compiled in the
# spacemit-llama.cpp tree.
#
#   IME2_BRIDGE_SRC=/work/.../tl_mahiru ./tools/riscv_llm/build_ime2_bridge.sh
#   -> build_riscv/libime2bridge.so
set -e

LL=${SPACEMIT_LLAMA:-/work/kubuds/hermes/spacemit-llama.cpp}
SRC=${IME2_BRIDGE_SRC:-/work/kubuds/hermes/tl_mahiru}
OUT=${OUT:-$PWD/build_riscv/libime2bridge.so}
HERE=$(cd "$(dirname "$0")" && pwd)
OBJ=$LL/build/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/spacemit

mkdir -p "$(dirname "$OUT")" /tmp/ime2_inc/ggml-spacemit
ln -sf "$LL/ggml/src/ggml-cpu/spacemit/repack.h" /tmp/ime2_inc/ggml-spacemit/repack.h
# ime2_bridge.cpp declares main_kernel but never uses it, and the vendor runtime
# does not export it: an empty stub keeps the link (and ctypes' RTLD_NOW) happy.
cat > /tmp/ime2_stub.c <<'EOF'
extern "C" int main_kernel(float *a, float *b, float *c) {
  (void)a; (void)b; (void)c; return 0;
}
EOF

g++ -O3 -fPIC -shared -march=rv64gcv -mabi=lp64d -fopenmp \
  -o "$OUT" "$SRC/riscv/ime2_bridge.cpp" "$HERE/ime2_gemm_m.cpp" /tmp/ime2_stub.c \
  "$OBJ/ime2_kernels.cpp.o" "$OBJ/rvv_kernels.cpp.o" "$OBJ/repack.cpp.o" \
  "$OBJ/ime1_kernels.cpp.o" \
  -I "$LL/ggml/include" -I "$LL/ggml/src" -I "$LL/ggml/src/ggml-cpu" \
  -I "$LL/ggml/src/ggml-cpu/spacemit" -I /tmp/ime2_inc \
  -Wl,--no-as-needed -L "$LL/build/bin" -lggml -lspert -lm

echo "wrote $OUT"
