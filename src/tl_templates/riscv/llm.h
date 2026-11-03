/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tl_templates/riscv/llm.h
 * \brief Transformer decode-step primitives for the RISC-V backend.
 *
 * These are the RVV 1.0 kernels the TileLang LLM driver (tilelang/riscv/llm)
 * calls from TIR.  Each one is a whole operator, not a tile: the tile-level
 * language is a poor fit for a GEMV whose weights are re-read from DRAM every
 * token, so the TIR program owns the *schedule* (which op runs where, with
 * which buffers and offsets) and these kernels own the inner loops.  This is
 * the same split TileLang uses on CUDA, where T.gemm expands to CUTLASS
 * templates.
 *
 * Numerics: weights are int8 with one fp16 scale per 32-element block
 * (llama.cpp Q8_0 layout, but stored as separate q/scale planes so the inner
 * loop does not chase interleaved metadata).  Activations stay fp32: the
 * dequantise-and-FMA form measures faster than quantising the activation on
 * this chip because the kernel is DRAM-bound, not FMA-bound.
 *
 * All kernels are VLEN-agnostic (vsetvl everywhere); nothing here assumes
 * VLEN=256.  The IME2 path is separate (ime2.h) because its register layout
 * is inherently width-specific.
 */
#ifndef TL_TEMPLATES_RISCV_LLM_H_
#define TL_TEMPLATES_RISCV_LLM_H_

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "tl_templates/riscv/riscv.h"

namespace tl {
namespace rv {

/* IEEE half -> float without depending on -mzfh (the baseline -march is
 * rv64gcv, which does not imply the scalar half extension). */
static inline float half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t man = h & 0x3ffu;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        exp--;
      }
      man &= 0x3ffu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7f800000u | (man << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out;
  memcpy(&out, &f, sizeof(out));
  return out;
}

static inline float hsum_f32m4(vfloat32m4_t v, size_t vl) {
  vfloat32m1_t s = __riscv_vfredusum_vs_f32m4_f32m1(
      v, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
  return __riscv_vfmv_f_s_f32m1_f32(s);
}

/* y[n] = sum_k x[k] * q[n,k] * scale[n,k/32]   (one row of a Q8 matvec) */
static inline void gemv_q8_impl(int N, int K, const uint8_t *q,
                                const uint8_t *sc, const float *x, float *y) {
  const int kb = K >> 5;
  const uint16_t *s = (const uint16_t *)sc;
  const int8_t *w = (const int8_t *)q;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (int n = 0; n < N; ++n) {
    const int8_t *wr = w + (size_t)n * K;
    const uint16_t *sr = s + (size_t)n * kb;
    size_t vl32 = __riscv_vsetvl_e32m4((size_t)32);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl32);
    for (int b = 0; b < kb; ++b) {
      const int off = b << 5;
      size_t vl = __riscv_vsetvl_e8m1((size_t)32);
      vint8m1_t wv = __riscv_vle8_v_i8m1(wr + off, vl);
      vint32m4_t wi = __riscv_vsext_vf4_i32m4(wv, vl);
      vfloat32m4_t wf = __riscv_vfcvt_f_x_v_f32m4(wi, vl);
      wf = __riscv_vfmul_vf_f32m4(wf, half_to_float(sr[b]), vl);
      vfloat32m4_t xv = __riscv_vle32_v_f32m4(x + off, vl);
      acc = __riscv_vfmacc_vv_f32m4(acc, wf, xv, vl);
    }
    y[n] = hsum_f32m4(acc, vl32);
  }
}

/* y = x * rsqrt(mean(x^2) + eps) * g   -- one row, K elements */
static inline void rmsnorm_impl(int K, const float *x, const float *g, float eps,
                                float *y) {
  size_t vmax = __riscv_vsetvlmax_e32m4();
  vfloat32m4_t vacc = __riscv_vfmv_v_f_f32m4(0.0f, vmax);
  int i = 0;
  while (i < K) {
    size_t vl = __riscv_vsetvl_e32m4((size_t)(K - i));
    vfloat32m4_t v = __riscv_vle32_v_f32m4(x + i, vl);
    vacc = __riscv_vfmacc_vv_f32m4(vacc, v, v, vl);
    i += (int)vl;
  }
  float ss = hsum_f32m4(vacc, vmax);
  float r = 1.0f / sqrtf(ss / (float)K + eps);
  i = 0;
  while (i < K) {
    size_t vl = __riscv_vsetvl_e32m4((size_t)(K - i));
    vfloat32m4_t v = __riscv_vle32_v_f32m4(x + i, vl);
    vfloat32m4_t gv = __riscv_vle32_v_f32m4(g + i, vl);
    vfloat32m4_t o = __riscv_vfmul_vv_f32m4(v, gv, vl);
    o = __riscv_vfmul_vf_f32m4(o, r, vl);
    __riscv_vse32_v_f32m4(y + i, o, vl);
    i += (int)vl;
  }
}

/* fp16 embedding table row -> fp32 activations */
static inline void embed_impl(int K, const uint16_t *tab, int tok_id, float *y) {
  const uint16_t *row = tab + (size_t)tok_id * K;
  for (int i = 0; i < K; ++i) {
    y[i] = half_to_float(row[i]);
  }
}

/* Per-head RMSNorm (Qwen3 q_norm/k_norm) over head_dim, one head. */
static inline void head_norm_impl(int hd, float *v, const float *g, float eps) {
  size_t vmax = __riscv_vsetvlmax_e32m4();
  vfloat32m4_t vacc = __riscv_vfmv_v_f_f32m4(0.0f, vmax);
  int i = 0;
  while (i < hd) {
    size_t vl = __riscv_vsetvl_e32m4((size_t)(hd - i));
    vfloat32m4_t x = __riscv_vle32_v_f32m4(v + i, vl);
    vacc = __riscv_vfmacc_vv_f32m4(vacc, x, x, vl);
    i += (int)vl;
  }
  float r = 1.0f / sqrtf(hsum_f32m4(vacc, vmax) / (float)hd + eps);
  i = 0;
  while (i < hd) {
    size_t vl = __riscv_vsetvl_e32m4((size_t)(hd - i));
    vfloat32m4_t x = __riscv_vle32_v_f32m4(v + i, vl);
    vfloat32m4_t gv = __riscv_vle32_v_f32m4(g + i, vl);
    __riscv_vse32_v_f32m4(
        v + i, __riscv_vfmul_vf_f32m4(__riscv_vfmul_vv_f32m4(x, gv, vl), r, vl), vl);
    i += (int)vl;
  }
}

/* NeoX-style RoPE over one head: pairs (d, d + hd/2).
 * cos/sin come from a precomputed [maxseq][hd/2] table: computing 57k
 * sinf/cosf per token would cost more than the GEMV it feeds. */
static inline void rope_head_impl(int hd, float *v, const float *cs,
                                  const float *sn) {
  const int half = hd >> 1;
  int d = 0;
  while (d < half) {
    size_t vl = __riscv_vsetvl_e32m4((size_t)(half - d));
    vfloat32m4_t x1 = __riscv_vle32_v_f32m4(v + d, vl);
    vfloat32m4_t x2 = __riscv_vle32_v_f32m4(v + d + half, vl);
    vfloat32m4_t c = __riscv_vle32_v_f32m4(cs + d, vl);
    vfloat32m4_t s = __riscv_vle32_v_f32m4(sn + d, vl);
    vfloat32m4_t o1 = __riscv_vfsub_vv_f32m4(__riscv_vfmul_vv_f32m4(x1, c, vl),
                                             __riscv_vfmul_vv_f32m4(x2, s, vl), vl);
    vfloat32m4_t o2 = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(x2, c, vl),
                                             __riscv_vfmul_vv_f32m4(x1, s, vl), vl);
    __riscv_vse32_v_f32m4(v + d, o1, vl);
    __riscv_vse32_v_f32m4(v + d + half, o2, vl);
    d += (int)vl;
  }
}

/* y = silu(g) * u, elementwise over K.
 * RVV 1.0 has no vector exp, so the exponential stays scalar; the loop is
 * still ~4x cheaper than the GEMV it follows. */
static inline void silu_mul_impl(int K, const float *g, const float *u, float *y) {
  int i = 0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (i = 0; i < K; ++i) {
    float x = g[i];
    y[i] = (x / (1.0f + expf(-x))) * u[i];
  }
}

static inline void add_impl(int K, float *dst, const float *src) {
  int i = 0;
  while (i < K) {
    size_t vl = __riscv_vsetvl_e32m4((size_t)(K - i));
    vfloat32m4_t a = __riscv_vle32_v_f32m4(dst + i, vl);
    vfloat32m4_t b = __riscv_vle32_v_f32m4(src + i, vl);
    __riscv_vse32_v_f32m4(dst + i, __riscv_vfadd_vv_f32m4(a, b, vl), vl);
    i += (int)vl;
  }
}

/* GQA decode attention for one token.
 *
 * q:      [nq][hd]
 * kcache: [nkv][maxseq][hd] (kvstride = maxseq*hd), row `pos` already written
 * out:    [nq][hd]
 * One thread per query head.  K/V for a KV head shared by several query
 * heads is re-read once per query head; that is the cache layout we have and
 * it keeps the kernel free of a transpose.
 */
static inline void attn_decode_impl(int nq, int nkv, int hd, int pos, int kvstride,
                                    float scale, const float *q,
                                    const float *kcache, const float *vcache,
                                    float *out) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (int h = 0; h < nq; ++h) {
    const int kvh = (int)((long long)h * nkv / nq);
    const float *qh = q + (size_t)h * hd;
    const float *kb = kcache + (size_t)kvh * kvstride;
    const float *vb = vcache + (size_t)kvh * kvstride;
    float m = -1e30f, l = 0.0f;
    float *o = out + (size_t)h * hd;
    for (int d = 0; d < hd; ++d) o[d] = 0.0f;
    for (int t = 0; t <= pos; ++t) {
      const float *k = kb + (size_t)t * hd;
      size_t vmax = __riscv_vsetvlmax_e32m4();
      vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vmax);
      int d = 0;
      while (d < hd) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(hd - d));
        vfloat32m4_t qv = __riscv_vle32_v_f32m4(qh + d, vl);
        vfloat32m4_t kv = __riscv_vle32_v_f32m4(k + d, vl);
        acc = __riscv_vfmacc_vv_f32m4(acc, qv, kv, vl);
        d += (int)vl;
      }
      float s = hsum_f32m4(acc, vmax) * scale;
      float m_new = (s > m) ? s : m;
      float e = expf(s - m_new);
      float corr = expf(m - m_new);
      l = l * corr + e;
      const float *v = vb + (size_t)t * hd;
      d = 0;
      while (d < hd) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(hd - d));
        vfloat32m4_t ov = __riscv_vle32_v_f32m4(o + d, vl);
        vfloat32m4_t vv = __riscv_vle32_v_f32m4(v + d, vl);
        ov = __riscv_vfmul_vf_f32m4(ov, corr, vl);
        ov = __riscv_vfmacc_vf_f32m4(ov, e, vv, vl);
        __riscv_vse32_v_f32m4(o + d, ov, vl);
        d += (int)vl;
      }
      m = m_new;
    }
    float inv = 1.0f / l;
    int d2 = 0;
    while (d2 < hd) {
      size_t vl = __riscv_vsetvl_e32m4((size_t)(hd - d2));
      vfloat32m4_t ov = __riscv_vle32_v_f32m4(o + d2, vl);
      __riscv_vse32_v_f32m4(o + d2, __riscv_vfmul_vf_f32m4(ov, inv, vl), vl);
      d2 += (int)vl;
    }
  }
}

static inline void copy_impl(int n, float *dst, const float *src) {
  int i = 0;
  while (i < n) {
    size_t vl = __riscv_vsetvl_e32m4((size_t)(n - i));
    __riscv_vse32_v_f32m4(dst + i, __riscv_vle32_v_f32m4(src + i, vl), vl);
    i += (int)vl;
  }
}

static inline void argmax_impl(int n, const float *x, int32_t *out) {
  int best = 0;
  float bv = x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] > bv) {
      bv = x[i];
      best = i;
    }
  }
  *out = best;
}

} // namespace rv
} // namespace tl

extern "C" {

static inline int32_t tl_riscv_gemv_q8(int32_t N, int32_t K, uint8_t *q,
                                       uint8_t *sc, float *x, float *y) {
  tl::rv::gemv_q8_impl(N, K, q, sc, x, y);
  return 0;
}

static inline int32_t tl_riscv_rmsnorm(int32_t K, float *x, uint8_t *g, float eps,
                                       float *y) {
  tl::rv::rmsnorm_impl(K, x, (const float *)g, eps, y);
  return 0;
}

static inline int32_t tl_riscv_head_norm(int32_t hd, float *v, uint8_t *g,
                                         float eps) {
  tl::rv::head_norm_impl(hd, v, (const float *)g, eps);
  return 0;
}

static inline int32_t tl_riscv_rope(int32_t hd, float *v, float *cs, float *sn) {
  tl::rv::rope_head_impl(hd, v, cs, sn);
  return 0;
}

static inline int32_t tl_riscv_embed(int32_t K, uint8_t *tab, int32_t tok,
                                     float *y) {
  tl::rv::embed_impl(K, (const uint16_t *)tab, tok, y);
  return 0;
}

static inline int32_t tl_riscv_silu_mul(int32_t K, float *g, float *u,
                                        float *y) {
  tl::rv::silu_mul_impl(K, g, u, y);
  return 0;
}

static inline int32_t tl_riscv_add(int32_t K, float *dst, float *src) {
  tl::rv::add_impl(K, dst, src);
  return 0;
}

/* Same, with the addend read straight out of the weight blob (Qwen2 qkv bias
 * is stored there as fp32). */
static inline int32_t tl_riscv_add_u8(int32_t K, float *dst, uint8_t *src) {
  tl::rv::add_impl(K, dst, (const float *)src);
  return 0;
}

static inline int32_t tl_riscv_attn_decode(int32_t nq, int32_t nkv, int32_t hd,
                                           int32_t pos, int32_t kvstride,
                                           float scale, float *q, float *kc,
                                           float *vc, float *out) {
  tl::rv::attn_decode_impl(nq, nkv, hd, pos, kvstride, scale, q, kc, vc, out);
  return 0;
}

static inline int32_t tl_riscv_copy(int32_t n, float *dst, float *src) {
  tl::rv::copy_impl(n, dst, src);
  return 0;
}

static inline int32_t tl_riscv_argmax(int32_t n, float *x, int32_t *out) {
  tl::rv::argmax_impl(n, x, out);
  return 0;
}

} // extern "C"

#endif // TL_TEMPLATES_RISCV_LLM_H_
