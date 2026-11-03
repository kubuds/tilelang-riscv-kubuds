/* Does one stream per row beat two (q plane + scale plane)?
 * layout A (current): all q rows contiguous, all scale rows in a separate plane.
 * layout B (fused):   each row = [q K bytes][scales K/16 bytes], stride K+K/16.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>

#include <riscv_vector.h>

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static inline float hsum(vfloat32m4_t v, size_t vl) {
  vfloat32m1_t s = __riscv_vfredusum_vs_f32m4_f32m1(v, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
  return __riscv_vfmv_f_s_f32m1_f32(s);
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

/* A: separate planes */
static void gemv_a(int N, int K, const uint8_t *q, const uint16_t *s,
                   const float *x, float *y) {
  const int kb = K >> 5;
#pragma omp parallel for schedule(static)
  for (int n = 0; n < N; ++n) {
    const uint8_t *qr = q + (size_t)n * K;
    const uint16_t *sr = s + (size_t)n * kb;
    size_t vl32 = __riscv_vsetvl_e32m4((size_t)32);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl32);
    for (int b = 0; b < kb; ++b) {
      const int off = b << 5;
      size_t vl = __riscv_vsetvl_e8m1((size_t)32);
      vint8m1_t wv = __riscv_vle8_v_i8m1((const int8_t *)qr + off, vl);
      vfloat32m4_t wf = __riscv_vfcvt_f_x_v_f32m4(__riscv_vsext_vf4_i32m4(wv, vl), vl);
      wf = __riscv_vfmul_vf_f32m4(wf, h2f(sr[b]), vl);
      acc = __riscv_vfmacc_vv_f32m4(acc, wf, __riscv_vle32_v_f32m4(x + off, vl), vl);
    }
    y[n] = hsum(acc, vl32);
  }
}

/* B: row-fused, stride K + K/16 bytes (scales right after the row) */
static void gemv_b(int N, int K, const uint8_t *q, const float *x, float *y) {
  const int kb = K >> 5;
  const int stride = K + (K >> 4);
#pragma omp parallel for schedule(static)
  for (int n = 0; n < N; ++n) {
    const uint8_t *qr = q + (size_t)n * stride;
    const uint16_t *sr = (const uint16_t *)(qr + K);
    size_t vl32 = __riscv_vsetvl_e32m4((size_t)32);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl32);
    for (int b = 0; b < kb; ++b) {
      const int off = b << 5;
      size_t vl = __riscv_vsetvl_e8m1((size_t)32);
      vint8m1_t wv = __riscv_vle8_v_i8m1((const int8_t *)qr + off, vl);
      vfloat32m4_t wf = __riscv_vfcvt_f_x_v_f32m4(__riscv_vsext_vf4_i32m4(wv, vl), vl);
      wf = __riscv_vfmul_vf_f32m4(wf, h2f(sr[b]), vl);
      acc = __riscv_vfmacc_vv_f32m4(acc, wf, __riscv_vle32_v_f32m4(x + off, vl), vl);
    }
    y[n] = hsum(acc, vl32);
  }
}

static uint8_t *alloc_big(size_t n) {
  void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) { perror("mmap"); exit(1); }
  madvise(p, n, MADV_HUGEPAGE);
  memset(p, 1, n);
  return (uint8_t *)p;
}

int main(void) {
  const int K = 1024, N = 2048, calls = 224;
  size_t rows = (size_t)calls * N;
  uint8_t *Qa = alloc_big(rows * K);
  uint16_t *S = (uint16_t *)alloc_big(rows * (K / 32) * 2);
  uint8_t *Qb = alloc_big(rows * (K + K / 16));
  for (size_t i = 0; i < rows * (size_t)(K + K / 16); ++i) Qb[i] = 1;
  float *x = (float *)malloc(K * 4);
  float *y = (float *)malloc(N * 4);
  for (int i = 0; i < K; ++i) x[i] = 0.01f * (i % 7);
  for (size_t i = 0; i < rows * (size_t)(K / 32); ++i) S[i] = 0x2C00;

  for (int rep = 0; rep < 2; ++rep) {
    double t0 = now();
    for (int c = 0; c < calls; ++c)
      gemv_a(N, K, Qa + (size_t)c * N * K, S + (size_t)c * N * (K / 32), x, y);
    double da = now() - t0;
    t0 = now();
    for (int c = 0; c < calls; ++c)
      gemv_b(N, K, Qb + (size_t)c * N * (K + K / 16), x, y);
    double db = now() - t0;
    printf("rep%d  plane-b  %.1f GB/s   fused  %.1f GB/s\n", rep,
           (double)rows * K / da / 1e9, (double)rows * K / db / 1e9);
  }
  return 0;
}
