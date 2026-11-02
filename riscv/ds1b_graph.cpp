// ds1b_graph.cpp - DeepSeek-R1-Distill-Qwen-1.5B (Qwen2 arch) decode in one spert launch.
#define GGML_COMMON_DECL_CPP
#include "ggml.h"
#include "ggml-common.h"
#include "ime_kernels.h"
#include "rvv_kernels.h"
#include "ggml-spacemit/repack.h"
#include "spert.hpp"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace ds {
enum { NL = 28, HID = 1536, NH = 12, NKV = 2, HD = 128, FF = 8960, VOCAB = 151936 };
struct Args {
    const void * wq[NL], * wk[NL], * wv[NL], * wo[NL], * wg[NL], * wu[NL], * wd[NL];
    const float * bq[NL], * bk[NL], * bv[NL];
    const float * an[NL], * fn[NL];
    const float * nf, * cosT, * sinT;
    const void * lm;
    float * x, * KC, * VC, * logits;
    int pos, maxT;
};

static inline void rmsnorm(const float * x, const float * w, float * y, int n, float eps) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += x[i] * x[i];
    float r = 1.0f / sqrtf(s / (float)n + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * r * w[i];
}

static inline void rope(float * v, int hd, int pos, const float * cosT, const float * sinT) {
    int half = hd / 2;
    for (int d = 0; d < half; ++d) {
        float a = v[d], b = v[d + half];
        float c = cosT[(size_t)pos * half + d], s = sinT[(size_t)pos * half + d];
        v[d] = a * c - b * s;
        v[d + half] = a * s + b * c;
    }
}

static float g_xn[4][HID];
static float g_q[NH * HD], g_k[NKV * HD], g_v[NKV * HD], g_att[NH * HD];
static float g_g[FF], g_u[FF], g_o[HID], g_d[HID];
static float g_sc[4][8192];
static uint8_t * g_qa = nullptr;
static size_t g_qa_bytes = 0;
}  // namespace ds

extern "C" int ds1b_graph_fwd(void * const * ptrs, int pos, int maxT) {
    using namespace ds;
    Args a{};
    int pi = 0;
    for (int l = 0; l < NL; ++l) {
        a.wq[l] = ptrs[pi++]; a.wk[l] = ptrs[pi++]; a.wv[l] = ptrs[pi++];
        a.wo[l] = ptrs[pi++]; a.wg[l] = ptrs[pi++]; a.wu[l] = ptrs[pi++]; a.wd[l] = ptrs[pi++];
        a.bq[l] = (const float *)ptrs[pi++];
        a.bk[l] = (const float *)ptrs[pi++];
        a.bv[l] = (const float *)ptrs[pi++];
        a.an[l] = (const float *)ptrs[pi++];
        a.fn[l] = (const float *)ptrs[pi++];
    }
    a.nf = (const float *)ptrs[pi++];
    a.cosT = (const float *)ptrs[pi++];
    a.sinT = (const float *)ptrs[pi++];
    a.lm = ptrs[pi++];
    a.x = (float *)ptrs[pi++];
    a.KC = (float *)ptrs[pi++];
    a.VC = (float *)ptrs[pi++];
    a.logits = (float *)ptrs[pi++];
    a.pos = pos;
    a.maxT = maxT;

    static spert::Stream * st = nullptr;
    if (!st) {
        spert::StreamConfig c;
        c.n_cores = 4;
        st = new spert::Stream(c);
        if (!st->valid()) return -1;
    }
    const size_t a_row = (size_t)(FF / 32) * spacemit_kernels::q8_blk_size(32, true);
    if (a_row * 4 > g_qa_bytes) {
        free(g_qa);
        g_qa_bytes = a_row * 4;
        g_qa = (uint8_t *)aligned_alloc(64, g_qa_bytes);
    }

    auto fut = st->launch(
        spert::Grid{4u},
        [](spert::Context * ctx, Args * a, uint8_t * qa, size_t a_row) {
            const uint32_t ith = ctx->program_id(0);
            uint8_t * qai = qa + (size_t)ith * a_row;

            auto gemv_b = [&](const void * w, const float * xs, float * y, const float * bias, int N, int K) {
                int per = (N + 3) / 4, n0 = ith * per, n1 = std::min(N, n0 + per);
                if (n0 >= n1) return;
                const size_t g = (size_t)(K / 32) * 576;
                const uint8_t * b = (const uint8_t *)w + (size_t)(n0 / 32) * g;
                spacemit_kernels::ime2::gemm_kernel_i8i4(32, qai, b, nullptr, y + n0, 1,
                                                         (size_t)(n1 - n0), K / 32, N);
                if (bias) for (int i = n0; i < n1; ++i) y[i] += bias[i];
            };

            for (int l = 0; l < NL; ++l) {
                rmsnorm(a->x, a->an[l], g_xn[ith], HID, 1e-6f);
                spacemit_kernels::rvv::quantize_a_row_i8(32, g_xn[ith], HID, qai);
                gemv_b(a->wq[l], g_xn[ith], g_q, a->bq[l], NH * HD, HID);
                gemv_b(a->wk[l], g_xn[ith], g_k, a->bk[l], NKV * HD, HID);
                gemv_b(a->wv[l], g_xn[ith], g_v, a->bv[l], NKV * HD, HID);
                ctx->sync();
                for (int h = ith; h < NH; h += 4) rope(g_q + h * HD, HD, a->pos, a->cosT, a->sinT);
                for (int h = ith; h < NKV; h += 4) {
                    rope(g_k + h * HD, HD, a->pos, a->cosT, a->sinT);
                    memcpy(a->KC + (((size_t)l * NKV + h) * a->maxT + a->pos) * HD, g_k + h * HD, HD * sizeof(float));
                    memcpy(a->VC + (((size_t)l * NKV + h) * a->maxT + a->pos) * HD, g_v + h * HD, HD * sizeof(float));
                }
                ctx->sync();
                for (int h = ith; h < NH; h += 4) {
                    const float * qh = g_q + h * HD;
                    const int kvh = h * NKV / NH;
                    const float * kh = a->KC + ((size_t)l * NKV + kvh) * a->maxT * HD;
                    const float * vh = a->VC + ((size_t)l * NKV + kvh) * a->maxT * HD;
                    float * sc = g_sc[ith];
                    float mx = -1e30f;
                    for (int t = 0; t <= a->pos; ++t) {
                        const float * kt = kh + (size_t)t * HD;
                        float s = 0.0f;
                        for (int dd = 0; dd < HD; ++dd) s += qh[dd] * kt[dd];
                        sc[t] = s * 0.08838834764831845f;
                        if (sc[t] > mx) mx = sc[t];
                    }
                    float sum = 0.0f;
                    for (int t = 0; t <= a->pos; ++t) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
                    float inv = 1.0f / sum;
                    float * oh = g_att + h * HD;
                    for (int dd = 0; dd < HD; ++dd) oh[dd] = 0.0f;
                    for (int t = 0; t <= a->pos; ++t) {
                        float w = sc[t] * inv;
                        const float * vt = vh + (size_t)t * HD;
                        for (int dd = 0; dd < HD; ++dd) oh[dd] += w * vt[dd];
                    }
                }
                ctx->sync();
                spacemit_kernels::rvv::quantize_a_row_i8(32, g_att, NH * HD, qai);
                gemv_b(a->wo[l], g_att, g_o, nullptr, HID, NH * HD);
                ctx->sync();
                {
                    int per = (HID + 3) / 4, n0 = ith * per, n1 = std::min((int)HID, n0 + per);
                    for (int i = n0; i < n1; ++i) a->x[i] += g_o[i];
                }
                ctx->sync();
                rmsnorm(a->x, a->fn[l], g_xn[ith], HID, 1e-6f);
                spacemit_kernels::rvv::quantize_a_row_i8(32, g_xn[ith], HID, qai);
                gemv_b(a->wg[l], g_xn[ith], g_g, nullptr, FF, HID);
                gemv_b(a->wu[l], g_xn[ith], g_u, nullptr, FF, HID);
                ctx->sync();
                {
                    int per = (FF + 3) / 4, n0 = ith * per, n1 = std::min((int)FF, n0 + per);
                    for (int i = n0; i < n1; ++i) g_g[i] = (g_g[i] / (1.0f + expf(-g_g[i]))) * g_u[i];
                }
                ctx->sync();
                spacemit_kernels::rvv::quantize_a_row_i8(32, g_g, FF, qai);
                gemv_b(a->wd[l], g_g, g_d, nullptr, HID, FF);
                ctx->sync();
                {
                    int per = (HID + 3) / 4, n0 = ith * per, n1 = std::min((int)HID, n0 + per);
                    for (int i = n0; i < n1; ++i) a->x[i] += g_d[i];
                }
                ctx->sync();
            }
            rmsnorm(a->x, a->nf, g_xn[ith], HID, 1e-6f);
            spacemit_kernels::rvv::quantize_a_row_i8(32, g_xn[ith], HID, qai);
            gemv_b(a->lm, g_xn[ith], a->logits, nullptr, VOCAB, HID);
            ctx->sync();
            if (ith == 0) {
                int best = 0;
                float bv = a->logits[0];
                for (int i = 1; i < VOCAB; ++i) if (a->logits[i] > bv) { bv = a->logits[i]; best = i; }
                a->logits[VOCAB] = (float)best;
            }
        },
        &a, g_qa, a_row);
    fut.sync();
    return 0;
}
