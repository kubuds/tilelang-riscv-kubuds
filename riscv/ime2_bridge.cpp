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
#include <cmath>
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

extern "C" int ime2_gemv_q4_w(const void * packed, const float * x, float * y, int N, int K, int nw) {
    static spert::Stream * streams[9] = {nullptr};
    static uint8_t * qa[9] = {nullptr};
    static size_t qa_bytes[9] = {0};
    if (nw < 1 || nw > 8) return -2;
    size_t a_row = (size_t)(K / 32) * spacemit_kernels::q8_blk_size(32, true);
    if (!streams[nw]) {
        spert::StreamConfig cfg;
        cfg.n_cores = (uint32_t)nw;
        streams[nw] = new spert::Stream(cfg);
        if (!streams[nw]->valid()) return -1;
    }
    if (a_row * nw > qa_bytes[nw]) {
        free(qa[nw]);
        qa_bytes[nw] = a_row * nw;
        qa[nw] = (uint8_t *)aligned_alloc(64, qa_bytes[nw]);
    }
    auto fut = streams[nw]->launch(
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
        qa[nw], x, a_row, packed, y, N, K, nw);
    fut.sync();
    return 0;
}

// GQA attention for one query token. q:(nh,hd), KC/VC:(nkv,maxT,hd), out:(nh,hd).
extern "C" int qwen_attn_fwd(const float * q, const float * KC, const float * VC,
                             float * out, int nh, int nkv, int hd, int pos, int maxT) {
    static float SC[16][8192];
    if (nh > 16 || pos >= 8192 || pos >= maxT) return -3;
    const float scale = 1.0f / sqrtf((float)hd);
#pragma omp parallel for num_threads(8) schedule(static)
    for (int h = 0; h < nh; ++h) {
        const float * qh = q + (size_t)h * hd;
        const float * kh = KC + (size_t)(h * nkv / nh) * maxT * hd;
        const float * vh = VC + (size_t)(h * nkv / nh) * maxT * hd;
        float * sc = SC[h];
        for (int t = 0; t <= pos; ++t) {
            const float * kt = kh + (size_t)t * hd;
            float s = 0.0f;
            for (int d = 0; d < hd; ++d) s += qh[d] * kt[d];
            sc[t] = s * scale;
        }
        float mx = -1e30f;
        for (int t = 0; t <= pos; ++t) mx = sc[t] > mx ? sc[t] : mx;
        float sum = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            float e = expf(sc[t] - mx);
            sc[t] = e;
            sum += e;
        }
        const float inv = 1.0f / sum;
        float * oh = out + (size_t)h * hd;
        for (int d = 0; d < hd; ++d) oh[d] = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            const float w = sc[t] * inv;
            const float * vt = vh + (size_t)t * hd;
            for (int d = 0; d < hd; ++d) oh[d] += w * vt[d];
        }
    }
    return 0;
}

static void rmsnorm_f32(const float * x, const float * w, float * y, int n, float eps) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += x[i] * x[i];
    float r = 1.0f / sqrtf(s / (float)n + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * r * w[i];
}

static void rope_head(float * v, int hd, int pos, const float * cosT, const float * sinT) {
    int half = hd / 2;
    for (int d = 0; d < half; ++d) {
        float a = v[d], b = v[d + half];
        float c = cosT[(size_t)pos * half + d], s = sinT[(size_t)pos * half + d];
        v[d] = a * c - b * s;
        v[d + half] = a * s + b * c;
    }
}

static void silu_mul_f32(const float * g, const float * u, float * o, int n) {
    for (int i = 0; i < n; ++i) o[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
}

// Qwen3-0.6B single layer, decode (m=1). HID=1024 NH=16 NKV=8 HD=128 FF=3072.
extern "C" int qwen_layer_fwd(
    float * x,
    const void * wq, const void * wk, const void * wv, const void * wo,
    const void * wg, const void * wu, const void * wd,
    const float * an, const float * fn, const float * qn, const float * kn,
    const float * cosT, const float * sinT,
    int pos, int maxT, float * KC, float * VC) {
    enum { HID = 1024, NH = 16, NKV = 8, HD = 128, FF = 3072 };
    static float xn[HID], q[NH * HD], k[NKV * HD], v[NKV * HD], att[NH * HD];
    static float g[FF], u[FF], o[HID], d[HID];

    rmsnorm_f32(x, an, xn, HID, 1e-6f);
    if (ime2_gemv_q4(wq, xn, q, NH * HD, HID)) return -1;
    if (ime2_gemv_q4(wk, xn, k, NKV * HD, HID)) return -1;
    if (ime2_gemv_q4(wv, xn, v, NKV * HD, HID)) return -1;

    for (int h = 0; h < NH; ++h) {
        rmsnorm_f32(q + h * HD, qn, q + h * HD, HD, 1e-6f);
        rope_head(q + h * HD, HD, pos, cosT, sinT);
    }
    for (int h = 0; h < NKV; ++h) {
        rmsnorm_f32(k + h * HD, kn, k + h * HD, HD, 1e-6f);
        rope_head(k + h * HD, HD, pos, cosT, sinT);
        for (int i = 0; i < HD; ++i) {
            KC[(size_t)h * maxT * HD + (size_t)pos * HD + i] = k[h * HD + i];
            VC[(size_t)h * maxT * HD + (size_t)pos * HD + i] = v[h * HD + i];
        }
    }
    if (qwen_attn_fwd(q, KC, VC, att, NH, NKV, HD, pos, maxT)) return -1;

    if (ime2_gemv_q4(wo, att, o, HID, NH * HD)) return -1;
    for (int i = 0; i < HID; ++i) x[i] += o[i];

    rmsnorm_f32(x, fn, xn, HID, 1e-6f);
    if (ime2_gemv_q4(wg, xn, g, FF, HID)) return -1;
    if (ime2_gemv_q4(wu, xn, u, FF, HID)) return -1;
    silu_mul_f32(g, u, g, FF);
    if (ime2_gemv_q4(wd, g, d, HID, FF)) return -1;
    for (int i = 0; i < HID; ++i) x[i] += d[i];
    return 0;
}
