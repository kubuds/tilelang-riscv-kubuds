/* Prefill kernel variants: the current M=8 Q8 GEMV needs 8 accumulators of
 * 32-bit lanes, which at VLEN=256 is 8 x m4 = 32 vector registers -- the whole
 * file, plus the weight vector and the converted weights.  It spills.  These
 * variants trade lane width for register headroom. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <riscv_vector.h>

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static inline float h2f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, f;
  if (exp == 0) {
    if (!man) f = sign;
    else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400u)) { man <<= 1; exp--; }
      man &= 0x3ffu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7f800000u | (man << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float o;
  memcpy(&o, &f, 4);
  return o;
}

#define GEMV_M8(NAME, VT, LMUL, VLMAX_EXPR, INLMUL)                                   \
  static void NAME(int N, int K, const uint8_t *q, const uint16_t *s,          \
                   const float *x, float *y) {                                \
    const int kb = K >> 5;                                                    \
    const int8_t *w = (const int8_t *)q;                                      \
    _Pragma("omp parallel for schedule(static)") for (int n = 0; n < N; ++n) { \
      const int8_t *wr = w + (size_t)n * K;                                   \
      const uint16_t *sr = s + (size_t)n * kb;                                \
      size_t vl = __riscv_vsetvl_e32##LMUL((size_t)32);                       \
      VT a0 = __riscv_vfmv_v_f_f32##LMUL(0.f, vl);                            \
      VT a1 = __riscv_vfmv_v_f_f32##LMUL(0.f, vl);                            \
      VT a2 = __riscv_vfmv_v_f_f32##LMUL(0.f, vl);                            \
      VT a3 = __riscv_vfmv_v_f_f32##LMUL(0.f, vl);                            \
      VT a4 = __riscv_vfmv_v_f_f32##LMUL(0.f, vl);                            \
      VT a5 = __riscv_vfmv_v_f_f32##LMUL(0.f, vl);                            \
      VT a6 = __riscv_vfmv_v_f_f32##LMUL(0.f, vl);                            \
      VT a7 = __riscv_vfmv_v_f_f32##LMUL(0.f, vl);                            \
      for (int b = 0; b < kb; ++b) {                                          \
        const int off = b << 5;                                               \
        for (int t = 0; t < (int)vl / VLMAX_EXPR; ++t) {                      \
          const int o2 = off + t * VLMAX_EXPR;                                \
          size_t v8 = __riscv_vsetvl_e8##INLMUL((size_t)VLMAX_EXPR);          \
          vint8##INLMUL##_t wv = __riscv_vle8_v_i8##INLMUL(wr + o2, v8);      \
          vfloat32##LMUL##_t wf = __riscv_vfcvt_f_x_v_f32##LMUL(              \
              __riscv_vsext_vf4_i32##LMUL(wv, v8), v8);                       \
          wf = __riscv_vfmul_vf_f32##LMUL(wf, h2f(sr[b]), v8);                \
          a0 = __riscv_vfmacc_vv_f32##LMUL(                                   \
              a0, wf, __riscv_vle32_v_f32##LMUL(x + o2, v8), v8);             \
          a1 = __riscv_vfmacc_vv_f32##LMUL(                                   \
              a1, wf, __riscv_vle32_v_f32##LMUL(x + K + o2, v8), v8);         \
          a2 = __riscv_vfmacc_vv_f32##LMUL(                                   \
              a2, wf, __riscv_vle32_v_f32##LMUL(x + 2 * K + o2, v8), v8);     \
          a3 = __riscv_vfmacc_vv_f32##LMUL(                                   \
              a3, wf, __riscv_vle32_v_f32##LMUL(x + 3 * K + o2, v8), v8);     \
          a4 = __riscv_vfmacc_vv_f32##LMUL(                                   \
              a4, wf, __riscv_vle32_v_f32##LMUL(x + 4 * K + o2, v8), v8);     \
          a5 = __riscv_vfmacc_vv_f32##LMUL(                                   \
              a5, wf, __riscv_vle32_v_f32##LMUL(x + 5 * K + o2, v8), v8);     \
          a6 = __riscv_vfmacc_vv_f32##LMUL(                                   \
              a6, wf, __riscv_vle32_v_f32##LMUL(x + 6 * K + o2, v8), v8);     \
          a7 = __riscv_vfmacc_vv_f32##LMUL(                                   \
              a7, wf, __riscv_vle32_v_f32##LMUL(x + 7 * K + o2, v8), v8);     \
        }                                                                     \
      }                                                                       \
      y[(size_t)0 * N + n] = hsum_##LMUL(a0, vl);                             \
      y[(size_t)1 * N + n] = hsum_##LMUL(a1, vl);                             \
      y[(size_t)2 * N + n] = hsum_##LMUL(a2, vl);                             \
      y[(size_t)3 * N + n] = hsum_##LMUL(a3, vl);                             \
      y[(size_t)4 * N + n] = hsum_##LMUL(a4, vl);                             \
      y[(size_t)5 * N + n] = hsum_##LMUL(a5, vl);                             \
      y[(size_t)6 * N + n] = hsum_##LMUL(a6, vl);                             \
      y[(size_t)7 * N + n] = hsum_##LMUL(a7, vl);                             \
    }                                                                         \
  }

static inline float hsum_m4(vfloat32m4_t v, size_t vl) {
  return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(
      v, __riscv_vfmv_v_f_f32m1(0.f, 1), vl));
}
static inline float hsum_m2(vfloat32m2_t v, size_t vl) {
  return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(
      v, __riscv_vfmv_v_f_f32m1(0.f, 1), vl));
}
static inline float hsum_m1(vfloat32m1_t v, size_t vl) {
  return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(
      v, __riscv_vfmv_v_f_f32m1(0.f, 1), vl));
}

GEMV_M8(gemv_m8_acc_m4, vfloat32m4_t, m4, 32, m1)
GEMV_M8(gemv_m8_acc_m2, vfloat32m2_t, m2, 16, mf2)
GEMV_M8(gemv_m8_acc_m1, vfloat32m1_t, m1, 8, mf4)

typedef void (*fn_t)(int, int, const uint8_t *, const uint16_t *, const float *,
                     float *);

int main(int argc, char **argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 2048;
  const int K = argc > 2 ? atoi(argv[2]) : 1024;
  const int reps = 200;
  uint8_t *q = (uint8_t *)aligned_alloc(4096, (size_t)N * K);
  uint16_t *s = (uint16_t *)aligned_alloc(4096, (size_t)N * (K / 32) * 2);
  float *x = (float *)aligned_alloc(64, (size_t)8 * K * 4);
  float *y = (float *)aligned_alloc(64, (size_t)8 * N * 4);
  memset(q, 1, (size_t)N * K);
  for (size_t i = 0; i < (size_t)N * (K / 32); ++i) s[i] = 0x2C00;
  for (int i = 0; i < 8 * K; ++i) x[i] = 0.01f * (i % 7);

  const char *names[3] = {"acc m4 (32 lane)", "acc m2 (16 lane)",
                          "acc m1 (8 lane)"};
  fn_t fns[3] = {gemv_m8_acc_m4, gemv_m8_acc_m2, gemv_m8_acc_m1};
  for (int v = 0; v < 3; ++v) {
    fns[v](N, K, q, s, x, y);
    double best = 1e9;
    for (int r = 0; r < 3; ++r) {
      double t0 = now();
      for (int i = 0; i < reps; ++i) fns[v](N, K, q, s, x, y);
      double dt = now() - t0;
      if (dt < best) best = dt;
    }
    printf("%-18s %7.1f ms   %.2f GB/s weights   %.0f token/s-equiv\n",
           names[v], best * 1e3 / reps, (double)N * K / (best / reps) / 1e9,
           8.0 / (best / reps) * 1e3);
  }
  return 0;
}
