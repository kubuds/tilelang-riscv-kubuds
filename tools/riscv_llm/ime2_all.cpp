// ime2_all.cpp - single C ABI bridge for K3 A100 IME2 Q4_0, M=1 and M=4..16.
//
// Why one file: spert's A100 worker pool is the 8-core cluster, and every
// spert::Stream reserves cores for its lifetime.  The first bridge created one
// stream per entry point (decode's M=1 and prefill's M=4 each), so once a
// second 8-worker stream existed the two oversubscribed the cluster and every
// launch came back garbage / hung (output "!!!!", prefill 30 s).  A single
// shared 8-worker stream serves both entry points, and the measured M=1 GEMV
// goes from ~20.7 GB/s (4 workers) to ~34 GB/s (8).
//
// Entry points (all extern "C", host preloads RTLD_GLOBAL like libopenblas):
//   int ime2_repack_q4(raw, packed, N, K)          - load-time Q4_0 -> IME2 tiles
//   int ime2_gemv_q4(packed, x, y, N, K)           - M=1 decode GEMV
//   int ime2_gemm_q4_m(packed, x, y, N, K, M)      - M=4/8/16 prefill GEMM
//
// Kernels are the vendor's (SpacemiT, MIT): rvv::quantize_a_{,4}row_i8 and
// ime2::gemm_kernel_i8i4, launched through spert (MIT).
#define GGML_COMMON_DECL_CPP
#include "ggml.h"
#include "ggml-common.h"
#include "ime_kernels.h"
#include "rvv_kernels.h"
#include "ggml-spacemit/repack.h"
#include "spert.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace {
constexpr int kNW = 8;   // A100 cores = 4 shared IME2 units
constexpr int kBlk = 32;

spert::Stream *shared_stream() {
  static spert::Stream *s = nullptr;
  if (!s) {
    spert::StreamConfig cfg;
    cfg.n_cores = (uint32_t)kNW;
    s = new spert::Stream(cfg);
  }
  return (s && s->valid()) ? s : nullptr;
}

// one interleaved 4-row A block = (K/32) * 152 bytes; 4-row variant writes
// [4 x f32 scale][4 x int16 asum][4 x 32 int8] per K block.
size_t quad_bytes(int K) {
  return (size_t)(K / kBlk) * spacemit_kernels::q8_blk_size(kBlk, true) * 4;
}

// shared activation scratch
uint8_t *qa_buf(size_t *cap, size_t need) {
  static uint8_t *qa = nullptr;
  static size_t bytes = 0;
  if (need > bytes) {
    free(qa);
    qa = (uint8_t *)aligned_alloc(64, need);
    bytes = need;
  }
  *cap = bytes;
  return qa;
}
}  // namespace

extern "C" int ime2_repack_q4(const void *raw, void *packed, int N, int K) {
  size_t bytes = (size_t)N * (K / 32) * 18;
  struct ggml_init_params ip = {bytes + 8 * 1024 * 1024, nullptr, false};
  struct ggml_context *ctx = ggml_init(ip);
  if (!ctx) return -1;
  struct ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
  if (!t) { ggml_free(ctx); return -2; }
  memcpy(t->data, raw, bytes);
  int rc = ggml::cpu::riscv64_spacemit::repack<block_q4_0, 32, 32>(t, raw, bytes);
  if (rc == 0) {
    size_t packed_bytes = (size_t)(N / 32) * (K / 32) * 576;
    memcpy(packed, t->data, packed_bytes);
  }
  ggml_free(ctx);
  return rc;
}

extern "C" int ime2_gemv_q4(const void *packed, const float *x, float *y,
                            int N, int K) {
  spert::Stream *stream = shared_stream();
  if (!stream) return -1;
  const size_t a_row = (size_t)(K / kBlk) * spacemit_kernels::q8_blk_size(kBlk, true);
  size_t cap;
  uint8_t *qa = qa_buf(&cap, a_row * kNW);

  // decode is DRAM-bound and its matrices are small (k/v N=512..1024), so 8
  // workers only add launch/sync overhead; run it on a 4-tile grid over the
  // shared 8-core stream.  Prefill (M=4..16) uses the full 8.
  const int nw = 4;
  auto fut = stream->launch(
      spert::Grid{(uint32_t)nw},
      [](spert::Context *c, uint8_t *qa, const float *x, size_t a_row,
         const void *w, float *y, int N, int K, int nw) {
        const uint32_t ith = c->program_id(0);
        const int nblk = N / 32;
        const int per = (nblk + nw - 1) / nw;            // 32-col blocks per worker
        const int n0 = ith * per * 32, n1 = std::min(N, n0 + per * 32);
        if (n0 >= n1) return;
        uint8_t *q = qa + (size_t)ith * a_row;
        spacemit_kernels::rvv::quantize_a_row_i8(kBlk, x, K, q);
        const size_t group = (size_t)(K / kBlk) * 576;
        const uint8_t *b = (const uint8_t *)w + (size_t)(n0 / 32) * group;
        spacemit_kernels::ime2::gemm_kernel_i8i4(kBlk, q, b, nullptr, y + n0, 1,
                                                 (size_t)(n1 - n0), K / kBlk,
                                                 (size_t)N);
      },
      qa, x, a_row, packed, y, N, K, nw);
  fut.sync();
  return 0;
}

extern "C" int ime2_gemm_q4_m(const void *packed, const float *x, float *y,
                              int N, int K, int M) {
  spert::Stream *stream = shared_stream();
  if (!stream) return -1;
  const int quads = (M + 3) / 4;
  const size_t qb = quad_bytes(K);
  size_t cap;
  uint8_t *qa = qa_buf(&cap, qb * (size_t)quads * kNW);

  auto fut = stream->launch(
      spert::Grid{(uint32_t)kNW},
      [](spert::Context *c, uint8_t *qa, const float *x, size_t qb,
         const void *w, float *y, int N, int K, int quads) {
        const uint32_t ith = c->program_id(0);
        const int nblk = N / 32;
        const int per = (nblk + kNW - 1) / kNW;          // 32-col blocks per worker
        const int n0 = ith * per * 32, n1 = std::min(N, n0 + per * 32);
        if (n0 >= n1) return;
        const size_t group = (size_t)(K / kBlk) * 576;
        const uint8_t *b = (const uint8_t *)w + (size_t)(n0 / 32) * group;
        for (int g = 0; g < quads; ++g) {
          uint8_t *qg = qa + (size_t)ith * qb * (size_t)quads + (size_t)g * qb;
          spacemit_kernels::rvv::quantize_a_4row_i8(kBlk, x + (size_t)g * 4 * K,
                                                    K, qg);
          spacemit_kernels::ime2::gemm_kernel_i8i4(
              kBlk, qg, b, nullptr, y + (size_t)g * 4 * N + n0, 4,
              (size_t)(n1 - n0), K / kBlk, (size_t)N);
        }
      },
      qa, x, qb, packed, y, N, K, quads);
  fut.sync();
  return quads * 4;
}
