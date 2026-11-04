/* Pure vector FMA throughput (no memory traffic): does the X100 cluster scale? */
#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <riscv_vector.h>

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(void) {
  const int iters = 2000000;
  double t0 = now();
#pragma omp parallel
  {
    size_t vl = __riscv_vsetvl_e32m2((size_t)16);
    vfloat32m2_t a = __riscv_vfmv_v_f_f32m2(1.0f, vl);
    vfloat32m2_t b = __riscv_vfmv_v_f_f32m2(1.0000001f, vl);
    vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2(0.5f, vl);
    vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(0.25f, vl);
    vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(0.125f, vl);
    vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(0.0625f, vl);
    for (int i = 0; i < iters; ++i) {
      c0 = __riscv_vfmacc_vv_f32m2(c0, a, b, vl);
      c1 = __riscv_vfmacc_vv_f32m2(c1, a, b, vl);
      c2 = __riscv_vfmacc_vv_f32m2(c2, a, b, vl);
      c3 = __riscv_vfmacc_vv_f32m2(c3, a, b, vl);
    }
    float s = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(__riscv_vfadd_vv_f32m2(c0, c1, vl),
                                         __riscv_vfmv_v_f_f32m1(0.0f, 1), vl));
    if (s == 12345.0f) printf("noop\n");
  }
  double dt = now() - t0;
  double gflop = (double)omp_get_max_threads() * iters * 4.0 * 16 * 2 / dt / 1e9;
  printf("threads=%d  %.1f ms  %.1f GFLOP/s\n", omp_get_max_threads(),
         dt * 1e3, gflop);
  return 0;
}
