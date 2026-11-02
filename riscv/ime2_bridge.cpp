// ime2_bridge.cpp - C ABI bridge for K3 A100 IME2 Q4_0 GEMV.
// Usage from TileLang: T.call_extern("int","ime2_gemv_q4", packed, x, y, N, K)
// Host must preload this .so with RTLD_GLOBAL (like libopenblas).
#define GGML_COMMON_DECL_CPP
#include "ggml.h"
#include "ggml-common.h"
#include "ime_kernels.h"
#include "rvv_kernels.h"
#include "ggml-spacemit/repack.h"
#include "spert.hpp"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

extern "C" int ime2_repack_q4(const void * raw, void * packed, int N, int K) {
    size_t bytes = (size_t)N * (K / 32) * 18;
    struct ggml_init_params ip = { bytes + 8 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) return -1;
    struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
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

extern "C" int ime2_gemv_q4(const void * packed, const float * x, float * y, int N, int K) {
    static spert::Stream * stream = nullptr;
    static uint8_t * qa = nullptr;
    static size_t qa_bytes = 0;
    const int nw = 4;
    size_t a_row = (size_t)(K / 32) * spacemit_kernels::q8_blk_size(32, true);
    if (!stream) {
        spert::StreamConfig cfg;
        cfg.n_cores = (uint32_t)nw;
        stream = new spert::Stream(cfg);
        if (!stream->valid()) return -1;
    }
    if (a_row * nw > qa_bytes) {
        free(qa);
        qa_bytes = a_row * nw;
        qa = (uint8_t *)aligned_alloc(64, qa_bytes);
    }
    auto fut = stream->launch(
        spert::Grid{(uint32_t)nw},
        [](spert::Context * c, uint8_t * qa, const float * x, size_t a_row,
           const void * w, float * y, int N, int K, int nw) {
            uint32_t ith = c->program_id(0);
            int per = (N + nw - 1) / nw, n0 = ith * per, n1 = std::min(N, n0 + per);
            if (n0 >= n1) return;
            uint8_t * q = qa + (size_t)ith * a_row;
            spacemit_kernels::rvv::quantize_a_row_i8(32, x, K, q);
            size_t group = (size_t)(K / 32) * 576;
            const uint8_t * b = (const uint8_t *)w + (size_t)(n0 / 32) * group;
            spacemit_kernels::ime2::gemm_kernel_i8i4(32, q, b, nullptr, y + n0, 1,
                                                     (size_t)(n1 - n0), K / 32, N);
        },
        qa, x, a_row, packed, y, N, K, nw);
    fut.sync();
    return 0;
}
