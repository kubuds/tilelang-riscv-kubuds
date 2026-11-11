// M=1 vs M=4 Q4_0 IME2: does the 4-row path produce the same numbers, and how
// many bytes per token does it save?
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

extern "C" {
int ime2_repack_q4(const void *raw, void *packed, int N, int K);
int ime2_gemv_q4(const void *packed, const float *x, float *y, int N, int K);
int ime2_gemm_q4_m(const void *packed, const float *x, float *y, int N, int K, int M);
}

static double now(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static uint32_t s = 7;
static uint32_t nx(void) { s = s * 1664525u + 1013904223u; return s; }

static void fill_q4(uint8_t *b, int N, int K) {
  size_t nb = (size_t)N * (K / 32);
  for (size_t i = 0; i < nb; ++i) {
    uint16_t sc = 0x2800 | (nx() & 0x07FF);
    memcpy(b + i * 18, &sc, 2);
    for (int j = 0; j < 16; ++j) b[i * 18 + 2 + j] = (uint8_t)(nx() & 0xFF);
  }
}

int main(int argc, char **argv) {
  int N = argc > 1 ? atoi(argv[1]) : 4096;
  int K = argc > 2 ? atoi(argv[2]) : 1024;
  int reps = argc > 3 ? atoi(argv[3]) : 30;
  int M = argc > 4 ? atoi(argv[4]) : 4;

  size_t raw_bytes = (size_t)N * (K / 32) * 18;
  size_t pk_bytes = (size_t)(N / 32) * (K / 32) * 576;
  uint8_t *raw = (uint8_t *)aligned_alloc(4096, raw_bytes);
  uint8_t *pk = (uint8_t *)aligned_alloc(4096, pk_bytes);
  float *x = (float *)aligned_alloc(64, (size_t)8 * K * 4);
  float *y1 = (float *)aligned_alloc(64, (size_t)N * 4);
  float *y4 = (float *)aligned_alloc(64, (size_t)8 * N * 4);
  fill_q4(raw, N, K);
  for (int i = 0; i < 8 * K; ++i) x[i] = 0.02f * (i % 11) - 0.1f;

  if (ime2_repack_q4(raw, pk, N, K) != 0) { printf("repack failed\n"); return 2; }

  // reference: four independent M=1 calls
  memset(y4, 0, (size_t)8 * N * 4);   // the kernel accumulates into C
  int rc = ime2_gemm_q4_m(pk, x, y4, N, K, M);
  float *ref = (float *)malloc((size_t)8 * N * 4);
  for (int m = 0; m < M; ++m) {
    ime2_gemv_q4(pk, x + (size_t)m * K, y1, N, K);
    memcpy(ref + (size_t)m * N, y1, (size_t)N * 4);
  }

  printf("  ref r0=%.4f %.4f %.4f  y4 r0=%.4f %.4f %.4f  r1=%.4f\n",
         ref[0], ref[1], ref[2], y4[0], y4[1], y4[2], y4[N]);
  double maxd = 0, maxy = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double a = y4[(size_t)m * N + n], b = ref[(size_t)m * N + n];
      maxd = fmax(maxd, fabs(a - b));
      maxy = fmax(maxy, fabs(b));
    }

  double best1 = 1e9, best4 = 1e9;
  for (int r = 0; r < 3; ++r) {
    double t0 = now();
    for (int i = 0; i < reps; ++i) {
      for (int m = 0; m < M; ++m) ime2_gemv_q4(pk, x + (size_t)m * K, y1, N, K);
    }
    best1 = fmin(best1, (now() - t0) / reps);
    t0 = now();
    for (int i = 0; i < reps; ++i) ime2_gemm_q4_m(pk, x, y4, N, K, M);
    best4 = fmin(best4, (now() - t0) / reps);
  }
  double bytes = (double)N * K * 0.5625;
  printf("N=%-6d K=%-5d  M=%d rc=%d  maxdiff %.3e (max|y| %.3f)\n", N, K, M, rc, maxd, maxy);
  printf("  %d x M=1: %.3f ms  (%.2f GB/s per %d tokens)\n",
         M, best1 * 1e3, M * bytes / best1 / 1e9, M);
  printf("  M=%d   : %.3f ms  (%.2f GB/s, %.2fx)\n", M, best4 * 1e3,
         bytes / best4 / 1e9, best1 / best4);
  return 0;
}
