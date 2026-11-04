/* m2 kernel, but one fused parallel region over (n-panel, m-block) instead of
 * one region per n-panel.  8-thread collapse of the current shape is what this
 * is trying to fix. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <riscv_vector.h>

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void gemm_m2_fused(int M, int N, int K, const float *A, int lda,
                          const float *B, int ldb, float *C, int ldc) {
  const int panel_n = 256, MR = 8;
  const int npan = (N + panel_n - 1) / panel_n;
  const int mblk = (M + MR - 1) / MR;
#pragma omp parallel for collapse(2) schedule(static)
  for (int p = 0; p < npan; ++p) {
    for (int ib = 0; ib < mblk; ++ib) {
      const int jp = p * panel_n;
      const int jn = (N - jp < panel_n) ? (N - jp) : panel_n;
      const int i0 = ib * MR;
      if (i0 + MR > M) continue;
      for (int j = 0; j < jn;) {
        size_t vl = __riscv_vsetvl_e32m2((size_t)(jn - j));
        const float *b = B + jp + j;
        vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2(0.f, vl);
        vfloat32m2_t a1 = __riscv_vfmv_v_f_f32m2(0.f, vl);
        vfloat32m2_t a2 = __riscv_vfmv_v_f_f32m2(0.f, vl);
        vfloat32m2_t a3 = __riscv_vfmv_v_f_f32m2(0.f, vl);
        vfloat32m2_t a4 = __riscv_vfmv_v_f_f32m2(0.f, vl);
        vfloat32m2_t a5 = __riscv_vfmv_v_f_f32m2(0.f, vl);
        vfloat32m2_t a6 = __riscv_vfmv_v_f_f32m2(0.f, vl);
        vfloat32m2_t a7 = __riscv_vfmv_v_f_f32m2(0.f, vl);
        const float *r0 = A + (size_t)(i0 + 0) * lda;
        const float *r1 = A + (size_t)(i0 + 1) * lda;
        const float *r2 = A + (size_t)(i0 + 2) * lda;
        const float *r3 = A + (size_t)(i0 + 3) * lda;
        const float *r4 = A + (size_t)(i0 + 4) * lda;
        const float *r5 = A + (size_t)(i0 + 5) * lda;
        const float *r6 = A + (size_t)(i0 + 6) * lda;
        const float *r7 = A + (size_t)(i0 + 7) * lda;
        for (int k = 0; k < K; ++k) {
          vfloat32m2_t bv = __riscv_vle32_v_f32m2(b + (size_t)k * ldb, vl);
          a0 = __riscv_vfmacc_vf_f32m2(a0, r0[k], bv, vl);
          a1 = __riscv_vfmacc_vf_f32m2(a1, r1[k], bv, vl);
          a2 = __riscv_vfmacc_vf_f32m2(a2, r2[k], bv, vl);
          a3 = __riscv_vfmacc_vf_f32m2(a3, r3[k], bv, vl);
          a4 = __riscv_vfmacc_vf_f32m2(a4, r4[k], bv, vl);
          a5 = __riscv_vfmacc_vf_f32m2(a5, r5[k], bv, vl);
          a6 = __riscv_vfmacc_vf_f32m2(a6, r6[k], bv, vl);
          a7 = __riscv_vfmacc_vf_f32m2(a7, r7[k], bv, vl);
        }
        __riscv_vse32_v_f32m2(C + (size_t)(i0 + 0) * ldc + jp + j, a0, vl);
        __riscv_vse32_v_f32m2(C + (size_t)(i0 + 1) * ldc + jp + j, a1, vl);
        __riscv_vse32_v_f32m2(C + (size_t)(i0 + 2) * ldc + jp + j, a2, vl);
        __riscv_vse32_v_f32m2(C + (size_t)(i0 + 3) * ldc + jp + j, a3, vl);
        __riscv_vse32_v_f32m2(C + (size_t)(i0 + 4) * ldc + jp + j, a4, vl);
        __riscv_vse32_v_f32m2(C + (size_t)(i0 + 5) * ldc + jp + j, a5, vl);
        __riscv_vse32_v_f32m2(C + (size_t)(i0 + 6) * ldc + jp + j, a6, vl);
        __riscv_vse32_v_f32m2(C + (size_t)(i0 + 7) * ldc + jp + j, a7, vl);
        j += (int)vl;
      }
    }
  }
}

int main(int argc, char **argv) {
  const int M = 512, N = 512, K = 512;
  float *A = (float *)aligned_alloc(64, (size_t)M * K * 4);
  float *B = (float *)aligned_alloc(64, (size_t)K * N * 4);
  float *C = (float *)aligned_alloc(64, (size_t)M * N * 4);
  for (int i = 0; i < M * K; ++i) A[i] = 0.001f * (i % 13);
  for (int i = 0; i < K * N; ++i) B[i] = 0.001f * (i % 17);
  gemm_m2_fused(M, N, K, A, K, B, N, C, N);
  double best = 1e9;
  for (int r = 0; r < 5; ++r) {
    double t0 = now();
    for (int i = 0; i < 5; ++i) gemm_m2_fused(M, N, K, A, K, B, N, C, N);
    double dt = (now() - t0) / 5;
    if (dt < best) best = dt;
  }
  printf("fused m2: %.2f ms  %.1f GFLOP/s\n", best * 1e3,
         2.0 * M * N * K / best / 1e9);
  return 0;
}
