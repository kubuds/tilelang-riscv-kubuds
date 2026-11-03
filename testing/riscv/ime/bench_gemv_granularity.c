/* Does call granularity limit the GEMV stream rate?
 * A: many small calls (2048x1024), like one layer's q/k/v/o/ffn
 * B: one large call (151936x1024), like lm_head
 * Same kernel, same buffer layout, cold weights. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tl_templates/riscv/riscv.h"

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static uint8_t *Q;
static uint16_t *S;

static void *alloc_weights(size_t nbytes) {
  void *p = aligned_alloc(4096, nbytes);
  if (!p) { perror("alloc"); exit(1); }
  memset(p, 1, nbytes);
  return p;
}

int main(void) {
  const int K = 1024;
  const size_t total = (size_t)512 * 1024 * 1024;      /* 512 MB of int8 */
  Q = (uint8_t *)alloc_weights(total);
  S = (uint16_t *)alloc_weights(total / 32 * 2);
  float *x = (float *)aligned_alloc(64, K * 4);
  float *y = (float *)aligned_alloc(64, 151936 * 4 + 4096);
  for (int i = 0; i < K; ++i) x[i] = 0.01f * (i % 7);
  for (size_t i = 0; i < total / 32; ++i) S[i] = 0x2C00;

  /* A: 224 calls of 2048 rows = 448 MB */
  double t0 = now();
  for (int c = 0; c < 224; ++c)
    tl_riscv_gemv_q8(2048, K, Q + (size_t)c * 2048 * K, (uint8_t *)S, x, y);
  double da = now() - t0;
  printf("A small calls: %d x 2048x1024 (%zu MB) %.1f ms  %.1f GB/s\n", 224,
         (size_t)224 * 2048 * K >> 20, da * 1e3,
         (double)224 * 2048 * K / da / 1e9);

  /* B: one call of 151936 rows = 155 MB */
  double t1 = now();
  tl_riscv_gemv_q8(151936, K, Q, (uint8_t *)S, x, y);
  double db = now() - t1;
  printf("B one big call: 151936x1024 (%zu MB) %.1f ms  %.1f GB/s\n",
         (size_t)151936 * K >> 20, db * 1e3, (double)151936 * K / db / 1e9);
  return 0;
}
