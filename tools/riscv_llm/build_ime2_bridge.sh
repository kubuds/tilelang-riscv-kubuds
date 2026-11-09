#!/bin/bash
# Build the IME2 Q4 bridge that the "ime2q4" blob layout calls.
#
# It needs (a) the vendor spert kernels, which are already compiled inside the
# spacemit-llama.cpp tree, and (b) the bridge source from the kubuds-riscv-mahiru
# branch (Apache-2.0).  Point IME2_BRIDGE_SRC at that checkout.
#
#   IME2_BRIDGE_SRC=/work/.../tl_mahiru ./tools/riscv_llm/build_ime2_bridge.sh
#   -> build_riscv/libime2bridge.so
set -e

LL=${SPACEMIT_LLAMA:-/work/kubuds/hermes/spacemit-llama.cpp}
SRC=${IME2_BRIDGE_SRC:-/work/kubuds/hermes/tl_mahiru}
OUT=${OUT:-$PWD/build_riscv/libime2bridge.so}
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
  -o "$OUT" "$SRC/riscv/ime2_bridge.cpp" /tmp/ime2_stub.c \
  "$OBJ/ime2_kernels.cpp.o" "$OBJ/rvv_kernels.cpp.o" "$OBJ/repack.cpp.o" \
  -I "$LL/ggml/include" -I "$LL/ggml/src" -I "$LL/ggml/src/ggml-cpu" \
  -I "$LL/ggml/src/ggml-cpu/spacemit" -I /tmp/ime2_inc \
  -Wl,--no-as-needed -L "$LL/build/bin" -lggml -lspert -lm

echo "wrote $OUT"
