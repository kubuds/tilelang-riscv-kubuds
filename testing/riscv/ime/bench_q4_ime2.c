// Per-operator cross-check: Q4_0 + IME2 (mahiru's path, vendor kernels via
// spert) against our int8 + RVV GEMV on the Qwen3-0.6B decode shapes.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

extern "C" {

int ime2_repack_q4(const void *raw, void *packed, int N, int K);
int ime2_gemv_q4(const void *packed, const float *x, float *y, int N, int K);
int ime2_gemv_q4_w(const void *packed, const float *x, float *y, int N, int K, int nw);
}

static double now(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static uint32_t rng = 12345;
static uint32_t nx(void) { rng = rng * 1664525u + 1013904223u; return rng; }

static void fill_q4_0(uint8_t *b, int N, int K) {
  size_t nb = (size_t)N * (K / 32);
  for (size_t i = 0; i < nb; ++i) {
    uint16_t sc = 0x2000 | (nx() & 0x0FFF);       // small positive half
    memcpy(b + i * 18, &sc, 2);
    for (int j = 0; j < 16; ++j) b[i * 18 + 2 + j] = (uint8_t)(nx() & 0xFF);
  }
}

int main(int argc, char **argv) {
  if (argc < 4) { printf("usage: %s N K [nw] [reps]\n", argv[0]); return 1; }
  int N = atoi(argv[1]), K = atoi(argv[2]);
  int nw = argc > 3 ? atoi(argv[3]) : 4;
  int reps = argc > 4 ? atoi(argv[4]) : 50;

  size_t raw_bytes = (size_t)N * (K / 32) * 18;
  size_t pk_bytes = (size_t)(N / 32) * (K / 32) * 576;
  uint8_t *raw = (uint8_t *)aligned_alloc(4096, raw_bytes);
  uint8_t *pk = (uint8_t *)aligned_alloc(4096, pk_bytes);
  float *x = (float *)aligned_alloc(64, (size_t)K * 4);
  float *y = (float *)aligned_alloc(64, (size_t)N * 4);
  fill_q4_0(raw, N, K);
  for (int i = 0; i < K; ++i) x[i] = 0.01f * (i % 5);

  double t0 = now();
  int rc = ime2_repack_q4(raw, pk, N, K);
  double trep = now() - t0;
  if (rc != 0) { printf("repack rc=%d\n", rc); return 2; }

  if (ime2_gemv_q4_w(pk, x, y, N, K, nw) != 0) { printf("launch failed\n"); return 3; }
  for (int i = 0; i < 10; ++i) ime2_gemv_q4_w(pk, x, y, N, K, nw);

  double best = 1e9;
  for (int r = 0; r < 3; ++r) {
    double a = now();
    for (int i = 0; i < reps; ++i) ime2_gemv_q4_w(pk, x, y, N, K, nw);
    double dt = (now() - a) / reps;
    if (dt < best) best = dt;
  }
  double gb = (double)N * K * 0.5625 / best / 1e9;
  printf("N=%-6d K=%-5d nw=%d  %.3f ms  %.2f GB/s (q4 bytes)  repack %.1f ms  y[0]=%.4f\n",
         N, K, nw, best * 1e3, gb, trep * 1e3, y[0]);
  return 0;
}
