/* M=8 GEMV vs eight M=1 GEMVs on the same weights: how much does prefill gain? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>

#include "gemv_m8.h"
#include "tl_templates/riscv/riscv.h"

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(void) {
  const int N = 2048, K = 1024, reps = 200;
  uint8_t *q = (uint8_t *)aligned_alloc(4096, (size_t)N * K);
  uint16_t *s = (uint16_t *)aligned_alloc(4096, (size_t)N * (K / 32) * 2);
  float *x8 = (float *)aligned_alloc(64, (size_t)8 * K * 4);
  float *y = (float *)aligned_alloc(64, (size_t)8 * N * 4);
  memset(q, 1, (size_t)N * K);
  for (size_t i = 0; i < (size_t)N * (K / 32); ++i) s[i] = 0x2C00;
  for (int i = 0; i < 8 * K; ++i) x8[i] = 0.01f * (i % 7);

  double t0 = now();
  for (int r = 0; r < reps; ++r)
    for (int m = 0; m < 8; ++m)
      tl_riscv_gemv_q8(N, K, q, (uint8_t *)s, x8 + m * K, y + m * N);
  double d1 = now() - t0;

  t0 = now();
  for (int r = 0; r < reps; ++r) gemv_q8_m8_impl(N, K, q, s, x8, y);
  double d8 = now() - t0;

  printf("8x M=1: %.1f ms   M=8: %.1f ms   speedup %.2fx\n", d1 * 1e3, d8 * 1e3, d1 / d8);
  printf("  M=8 weights %.0f MB in %.1f ms = %.1f GB/s\n", (double)N * K / 1e6,
         d8 * 1e3, (double)reps * N * K / d8 / 1e9);
  return 0;
}
