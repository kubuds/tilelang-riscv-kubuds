/* Can the K=3072 down_proj be fixed by slicing K at the call level instead of
 * inside the kernel?  Each slice keeps an 8x1024 activation tile (32 KB) in L1
 * and costs exactly one OpenMP region, which is what the earlier in-kernel
 * K-blocking got wrong (6 regions per call, 629 -> 842 ms end to end).
 *
 * Weights and activations are both row-strided at the full K, so the same
 * kernel serves a slice by advancing the base pointer and passing strides.
 */
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

static inline float hsum_m1(vfloat32m1_t v, size_t vl) {
  return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(
      v, __riscv_vfmv_v_f_f32m1(0.f, 1), vl));
}

static inline float h2f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, f;
  if (exp == 0) {
    if (!man) f = sign;
    else { exp = 127 - 15 + 1; while (!(man & 0x400u)) { man <<= 1; exp--; } man &= 0x3ffu; f = sign | (exp << 23) | (man << 13); }
  } else if (exp == 31) f = sign | 0x7f800000u | (man << 13);
  else f = sign | ((exp - 15 + 127) << 23) | (man << 13);
  float o; memcpy(&o, &f, 4); return o;
}

/* One slice: K elements starting at (x + koff) and (w + n*ws + koff). */
static void gemv_slice(int N, int K, int ws, int xs, const uint8_t *q,
                       const uint16_t *s, const float *x, float *y, int beta) {
  const int kb = K >> 5;
  const int8_t *w = (const int8_t *)q;
  if (!beta) {
    for (int m = 0; m < 8; ++m)
      memset(y + (size_t)m * N, 0, (size_t)N * sizeof(float));
  }
#pragma omp parallel for schedule(static)
  for (int n = 0; n < N; ++n) {
    const int8_t *wr = w + (size_t)n * ws;
    const uint16_t *sr = s + (size_t)n * (ws / 32);
    size_t vl = __riscv_vsetvl_e32m1((size_t)8);
    vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
    vfloat32m1_t a1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
    vfloat32m1_t a2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
    vfloat32m1_t a3 = __riscv_vfmv_v_f_f32m1(0.f, vl);
    vfloat32m1_t a4 = __riscv_vfmv_v_f_f32m1(0.f, vl);
    vfloat32m1_t a5 = __riscv_vfmv_v_f_f32m1(0.f, vl);
    vfloat32m1_t a6 = __riscv_vfmv_v_f_f32m1(0.f, vl);
    vfloat32m1_t a7 = __riscv_vfmv_v_f_f32m1(0.f, vl);
    for (int b = 0; b < kb; ++b) {
      const int off = b << 5;
      const float sc = h2f(sr[b]);
      for (int t = 0; t < 4; ++t) {
        const int o2 = off + (t << 3);
        size_t v8 = __riscv_vsetvl_e8mf4((size_t)8);
        vint8mf4_t wv = __riscv_vle8_v_i8mf4(wr + o2, v8);
        vfloat32m1_t wf =
            __riscv_vfcvt_f_x_v_f32m1(__riscv_vsext_vf4_i32m1(wv, v8), v8);
        wf = __riscv_vfmul_vf_f32m1(wf, sc, v8);
        a0 = __riscv_vfmacc_vv_f32m1(a0, wf, __riscv_vle32_v_f32m1(x + o2, v8), v8);
        a1 = __riscv_vfmacc_vv_f32m1(a1, wf, __riscv_vle32_v_f32m1(x + xs + o2, v8), v8);
        a2 = __riscv_vfmacc_vv_f32m1(a2, wf, __riscv_vle32_v_f32m1(x + 2 * xs + o2, v8), v8);
        a3 = __riscv_vfmacc_vv_f32m1(a3, wf, __riscv_vle32_v_f32m1(x + 3 * xs + o2, v8), v8);
        a4 = __riscv_vfmacc_vv_f32m1(a4, wf, __riscv_vle32_v_f32m1(x + 4 * xs + o2, v8), v8);
        a5 = __riscv_vfmacc_vv_f32m1(a5, wf, __riscv_vle32_v_f32m1(x + 5 * xs + o2, v8), v8);
        a6 = __riscv_vfmacc_vv_f32m1(a6, wf, __riscv_vle32_v_f32m1(x + 6 * xs + o2, v8), v8);
        a7 = __riscv_vfmacc_vv_f32m1(a7, wf, __riscv_vle32_v_f32m1(x + 7 * xs + o2, v8), v8);
      }
    }
    y[(size_t)0 * N + n] += hsum_m1(a0, vl);
    y[(size_t)1 * N + n] += hsum_m1(a1, vl);
    y[(size_t)2 * N + n] += hsum_m1(a2, vl);
    y[(size_t)3 * N + n] += hsum_m1(a3, vl);
    y[(size_t)4 * N + n] += hsum_m1(a4, vl);
    y[(size_t)5 * N + n] += hsum_m1(a5, vl);
    y[(size_t)6 * N + n] += hsum_m1(a6, vl);
    y[(size_t)7 * N + n] += hsum_m1(a7, vl);
  }
}

int main(int argc, char **argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 1024;
  const int K = argc > 2 ? atoi(argv[2]) : 3072;
  const int SLICE = argc > 3 ? atoi(argv[3]) : 1024;
  const int reps = 50;
  uint8_t *q = (uint8_t *)aligned_alloc(4096, (size_t)N * K);
  uint16_t *s = (uint16_t *)aligned_alloc(4096, (size_t)N * (K / 32) * 2);
  float *x = (float *)aligned_alloc(64, (size_t)8 * K * 4);
  float *y = (float *)aligned_alloc(64, (size_t)8 * N * 4);
  memset(q, 1, (size_t)N * K);
  for (size_t i = 0; i < (size_t)N * (K / 32); ++i) s[i] = 0x2C00;
  for (int i = 0; i < 8 * K; ++i) x[i] = 0.01f * (i % 7);

  /* whole-K, one call (this is what the engine does today via gemv_q8_m8) */
  gemv_slice(N, K, K, K, q, s, x, y, 0);
  double best1 = 1e9;
  for (int r = 0; r < 3; ++r) {
    double t0 = now();
    for (int i = 0; i < reps; ++i) gemv_slice(N, K, K, K, q, s, x, y, 0);
    double dt = (now() - t0) / reps;
    if (dt < best1) best1 = dt;
  }

  /* sliced: each call is one region with a 32 KB activation tile */
  double best2 = 1e9;
  for (int r = 0; r < 3; ++r) {
    double t0 = now();
    for (int i = 0; i < reps; ++i) {
      int first = 1;
      for (int k0 = 0; k0 < K; k0 += SLICE) {
        int kl = (K - k0 < SLICE) ? (K - k0) : SLICE;
        gemv_slice(N, kl, K, K, q + k0, s + k0 / 32, x + k0, y, first);
        first = 0;
      }
    }
    double dt = (now() - t0) / reps;
    if (dt < best2) best2 = dt;
  }

  printf("N=%d K=%d  whole %.2f ms (%.2f GB/s)  sliced@%d %.2f ms (%.2f GB/s)  speedup %.2fx\n",
         N, K, best1 * 1e3, (double)N * K / best1 / 1e9, SLICE, best2 * 1e3,
         (double)N * K / best2 / 1e9, best1 / best2);
  return 0;
}
