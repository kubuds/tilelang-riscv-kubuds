#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>

static double now(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 512;
  int reps = argc > 2 ? atoi(argv[2]) : 10;
  float *a = aligned_alloc(64, (size_t)n * n * 4);
  float *b = aligned_alloc(64, (size_t)n * n * 4);
  float *c = aligned_alloc(64, (size_t)n * n * 4);
  for (int i = 0; i < n * n; ++i) { a[i] = 0.001f * (i % 97); b[i] = 0.001f * (i % 89); }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, a, n,
              b, n, 0.0f, c, n);
  double best = 1e9;
  for (int r = 0; r < 3; ++r) {
    double t0 = now();
    for (int i = 0; i < reps; ++i)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, a,
                  n, b, n, 0.0f, c, n);
    double dt = (now() - t0) / reps;
    if (dt < best) best = dt;
  }
  printf("cblas %d^3  %.2f ms  %.1f GFLOP/s\n", n, best * 1e3,
         2.0 * n * n * n / best / 1e9);
  return 0;
}
