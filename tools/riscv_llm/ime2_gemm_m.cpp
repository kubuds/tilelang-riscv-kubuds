// M=4 Q4_0 GEMM over the vendor IME2 kernels, for prefill.
//
// The M=1 path (ime2_bridge.cpp) re-reads every weight row for each token, which
// is why our 0.6B prefill is 25 ms/token against llama.cpp's 3.3.  The vendor
// dispatcher already has an m4 fast path; what differs is the activation
// packing: gemm_kernel_i8i4_m4 reads, per K-block, [4 x f32 scale][4 x int16
// asum][4 x 32 int8] = 152 bytes, which is exactly what
// rvv::quantize_a_4row_i8 writes (a_nrow_block_stride = q8_blk_size(32,true)*4).
//
//   int ime2_gemm_q4_m(const void *packed, const float *x, float *y,
//                      int N, int K, int M)
#include "ggml.h"
#include "ggml-common.h"
#include "ime_kernels.h"
#include "rvv_kernels.h"
#include "ggml-spacemit/repack.h"
#include "spert.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace {
constexpr int kNW = 4;      // A100 workers
constexpr int kBlk = 32;    // Q8_0/Q4_0 block length
}

extern "C" int ime2_gemm_q4_m(const void *packed, const float *x, float *y,
                              int N, int K, int M) {
  static spert::Stream *stream = nullptr;
  static uint8_t *qa = nullptr;
  static size_t qa_bytes = 0;

  const int quads = (M + 3) / 4;       // M is processed 4 rows at a time
  const int rows = 4;
  const size_t a_row = (size_t)(K / kBlk) * spacemit_kernels::q8_blk_size(kBlk, true);
  const size_t a_bytes = a_row * 4 * (size_t)quads;   // one interleaved quad per 4 rows

  if (!stream) {
    spert::StreamConfig cfg;
    cfg.n_cores = (uint32_t)kNW;
    stream = new spert::Stream(cfg);
    if (!stream->valid()) return -1;
  }
  if (a_bytes * kNW > qa_bytes) {
    free(qa);
    qa_bytes = a_bytes * kNW;
    qa = (uint8_t *)aligned_alloc(64, qa_bytes);
  }

  auto fut = stream->launch(
      spert::Grid{(uint32_t)kNW},
      [](spert::Context *c, uint8_t *qa, const float *x, size_t a_bytes,
         const void *w, float *y, int N, int K, int rows, int quads) {
        const uint32_t ith = c->program_id(0);
        const int per = (N + kNW - 1) / kNW;
        const int n0 = ith * per, n1 = std::min(N, n0 + per);
        if (n0 >= n1) return;
        uint8_t *q = qa + (size_t)ith * a_bytes;
        const size_t group = (size_t)(K / kBlk) * 576;
        const uint8_t *b = (const uint8_t *)w + (size_t)(n0 / 32) * group;
        const size_t quad_bytes = a_bytes / (size_t)quads;
        for (int g = 0; g < quads; ++g) {
          uint8_t *qg = q + (size_t)g * quad_bytes;
          spacemit_kernels::rvv::quantize_a_4row_i8(
              kBlk, x + (size_t)g * 4 * K, K, qg);
          spacemit_kernels::ime2::gemm_kernel_i8i4(
              kBlk, qg, b, nullptr, y + (size_t)g * 4 * N + n0, (size_t)rows,
              (size_t)(n1 - n0), K / kBlk, (size_t)N);
        }
      },
      qa, x, a_bytes, packed, y, N, K, rows, quads);
  fut.sync();
  return quads * 4;
}
