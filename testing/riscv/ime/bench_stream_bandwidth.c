/* Streaming read bandwidth, accumulate-only (one reduction at the end). */
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

int main(void) {
  size_t n = (size_t)1024 * 1024 * 1024;
  uint8_t *p = (uint8_t *)mmap(NULL, n, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  madvise(p, n, MADV_HUGEPAGE);
  memset(p, 1, n);

  double t0 = now();
  size_t total = 0;
#pragma omp parallel reduction(+ : total)
  {
    vuint8m1_t a0 = __riscv_vmv_v_x_u8m1(0, 32);
    vuint8m1_t a1 = __riscv_vmv_v_x_u8m1(0, 32);
    vuint8m1_t a2 = __riscv_vmv_v_x_u8m1(0, 32);
    vuint8m1_t a3 = __riscv_vmv_v_x_u8m1(0, 32);
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i += 128) {
      a0 = __riscv_vadd_vv_u8m1(a0, __riscv_vle8_v_u8m1(p + i, 32), 32);
      a1 = __riscv_vadd_vv_u8m1(a1, __riscv_vle8_v_u8m1(p + i + 32, 32), 32);
      a2 = __riscv_vadd_vv_u8m1(a2, __riscv_vle8_v_u8m1(p + i + 64, 32), 32);
      a3 = __riscv_vadd_vv_u8m1(a3, __riscv_vle8_v_u8m1(p + i + 96, 32), 32);
    }
    vuint8m1_t s = __riscv_vadd_vv_u8m1(__riscv_vadd_vv_u8m1(a0, a1, 32),
                                        __riscv_vadd_vv_u8m1(a2, a3, 32), 32);
    total += __riscv_vmv_x_s_u8m1_u8(
        __riscv_vredsum_vs_u8m1_u8m1(s, __riscv_vmv_v_x_u8m1(0, 1), 32));
  }
  double dt = now() - t0;
  printf("stream read 1 GB: %.1f ms  %.1f GB/s\n", dt * 1e3,
         (double)n / dt / 1e9);
  return 0;
}
