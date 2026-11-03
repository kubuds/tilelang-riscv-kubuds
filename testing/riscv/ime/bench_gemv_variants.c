/* GEMV streaming variants: baseline / prefetch / unroll / hugepage / both.
 * Cold weights, 448 MB, same arithmetic as the engine kernel. */
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
    else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400u)) { man <<= 1; exp--; }
      man &= 0x3ffu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) f = sign | 0x7f800000u | (man << 13);
  else f = sign | ((exp - 15 + 127) << 23) | (man << 13);
  float o;
  memcpy(&o, &f, 4);
  return o;
}

#define GEMV_BODY(NAME, PREFETCH, UNROLL)                                     \
  static void NAME(int N, int K, const uint8_t *q, const uint16_t *s,         \
                   const float *x, float *y) {                                \
    const int kb = K >> 5;                                                    \
    _Pragma("omp parallel for schedule(static)")                              \
    for (int n = 0; n < N; ++n) {                                             \
      const uint8_t *qr = q + (size_t)n * K;                                  \
      const uint16_t *sr = s + (size_t)n * kb;                                \
      size_t vl32 = __riscv_vsetvl_e32m4((size_t)32);                         \
      vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl32);                  \
      for (int b = 0; b < kb; ++b) {                                          \
        if (PREFETCH) __builtin_prefetch(qr + ((b + 4) << 5), 0, 0);          \
        const int off = b << 5;                                               \
        size_t vl = __riscv_vsetvl_e8m1((size_t)32);                          \
        vint8m1_t wv = __riscv_vle8_v_i8m1((const int8_t *)qr + off, vl);     \
        vint32m4_t wi = __riscv_vsext_vf4_i32m4(wv, vl);                      \
        vfloat32m4_t wf = __riscv_vfcvt_f_x_v_f32m4(wi, vl);                  \
        wf = __riscv_vfmul_vf_f32m4(wf, h2f(sr[b]), vl);                      \
        vfloat32m4_t xv = __riscv_vle32_v_f32m4(x + off, vl);                 \
        acc = __riscv_vfmacc_vv_f32m4(acc, wf, xv, vl);                       \
      }                                                                       \
      y[n] = hsum(acc, vl32);                                                 \
    }                                                                         \
  }

GEMV_BODY(gemv_base, 0, 0)
GEMV_BODY(gemv_pf, 1, 0)

static uint8_t *alloc_big(size_t n, int huge) {
  void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) { perror("mmap"); exit(1); }
  if (huge) madvise(p, n, MADV_HUGEPAGE);
  memset(p, 1, n);
  return (uint8_t *)p;
}

int main(int argc, char **argv) {
  const int K = 1024, N = 2048, calls = 224;
  size_t total = (size_t)calls * N * K;
  int huge = argc > 1 && atoi(argv[1]);
  uint8_t *Q = alloc_big(total, huge);
  uint16_t *S = (uint16_t *)alloc_big(total / 32 * 2, huge);
  float *x = (float *)malloc(K * 4);
  float *y = (float *)malloc(N * 4);
  for (int i = 0; i < K; ++i) x[i] = 0.01f * (i % 7);
  for (size_t i = 0; i < total / 32; ++i) S[i] = 0x2C00;

  /* touch the scale plane once so it is resident */
  for (int i = 0; i < 2048; ++i) y[i] = y[i];

  double t0 = now();
  for (int c = 0; c < calls; ++c)
    gemv_base(N, K, Q + (size_t)c * N * K, S, x, y);
  double d0 = now() - t0;
  printf("base     %zu MB  %.1f ms  %.1f GB/s\n", total >> 20, d0 * 1e3,
         (double)total / d0 / 1e9);

  t0 = now();
  for (int c = 0; c < calls; ++c)
    gemv_pf(N, K, Q + (size_t)c * N * K, S, x, y);
  double d1 = now() - t0;
  printf("prefetch %zu MB  %.1f ms  %.1f GB/s\n", total >> 20, d1 * 1e3,
         (double)total / d1 / 1e9);

  t0 = now();
  for (int c = 0; c < calls; ++c)
    gemv_base(N, K, Q + (size_t)c * N * K, S, x, y);
  double d2 = now() - t0;
  printf("base(2)  %zu MB  %.1f ms  %.1f GB/s\n", total >> 20, d2 * 1e3,
         (double)total / d2 / 1e9);
  printf("huge=%d\n", huge);
  return 0;
}
