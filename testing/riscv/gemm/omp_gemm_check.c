/* Does the GEMM's parallel loop actually spread across threads? */
#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(void) {
  const int M = 512, N = 512;
  double *C = (double *)aligned_alloc(64, (size_t)M * N * 8);
  static int hits[64];

  double t0 = now();
#pragma omp parallel for schedule(static)
  for (int i0 = 0; i0 <= M - 8; i0 += 8) {
    int tid = omp_get_thread_num();
#pragma omp atomic
    hits[tid]++;
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < 8; ++i) C[(size_t)(i0 + i) * N + j] += 1.0;
  }
  double dt = now() - t0;
  printf("threads=%d time=%.1f ms  per-thread chunks:", omp_get_max_threads(),
         dt * 1e3);
  for (int i = 0; i < omp_get_max_threads(); ++i) printf(" %d", hits[i]);
  printf("\n");
  return 0;
}
