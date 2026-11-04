/* fp32 GEMM variants: the current template reads B with a 2 KB stride in 8-lane
 * chunks (m1), so a 256-wide panel is swept 32 times.  These variants widen the
 * column vector and re-balance the row tile. */
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

/* V0: current template -- MR=8, e32m1 (8 lanes at VLEN=256). */
static void gemm_m1_mr8(int M, int N, int K, const float *A, int lda,
                        const float *B, int ldb, float *C, int ldc) {
  const int panel_n = 256, MR = 8;
  for (int jp = 0; jp < N; jp += panel_n) {
    const int jn = (N - jp < panel_n) ? (N - jp) : panel_n;
#pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 <= M - MR; i0 += MR) {
      for (int j = 0; j < jn;) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(jn - j));
        const float *b = B + jp + j;
        vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
        vfloat32m1_t a1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
        vfloat32m1_t a2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
        vfloat32m1_t a3 = __riscv_vfmv_v_f_f32m1(0.f, vl);
        vfloat32m1_t a4 = __riscv_vfmv_v_f_f32m1(0.f, vl);
        vfloat32m1_t a5 = __riscv_vfmv_v_f_f32m1(0.f, vl);
        vfloat32m1_t a6 = __riscv_vfmv_v_f_f32m1(0.f, vl);
        vfloat32m1_t a7 = __riscv_vfmv_v_f_f32m1(0.f, vl);
        const float *r0 = A + (size_t)(i0 + 0) * lda;
        const float *r1 = A + (size_t)(i0 + 1) * lda;
        const float *r2 = A + (size_t)(i0 + 2) * lda;
        const float *r3 = A + (size_t)(i0 + 3) * lda;
        const float *r4 = A + (size_t)(i0 + 4) * lda;
        const float *r5 = A + (size_t)(i0 + 5) * lda;
        const float *r6 = A + (size_t)(i0 + 6) * lda;
        const float *r7 = A + (size_t)(i0 + 7) * lda;
        for (int k = 0; k < K; ++k) {
          vfloat32m1_t bv = __riscv_vle32_v_f32m1(b + (size_t)k * ldb, vl);
          a0 = __riscv_vfmacc_vf_f32m1(a0, r0[k], bv, vl);
          a1 = __riscv_vfmacc_vf_f32m1(a1, r1[k], bv, vl);
          a2 = __riscv_vfmacc_vf_f32m1(a2, r2[k], bv, vl);
          a3 = __riscv_vfmacc_vf_f32m1(a3, r3[k], bv, vl);
          a4 = __riscv_vfmacc_vf_f32m1(a4, r4[k], bv, vl);
          a5 = __riscv_vfmacc_vf_f32m1(a5, r5[k], bv, vl);
          a6 = __riscv_vfmacc_vf_f32m1(a6, r6[k], bv, vl);
          a7 = __riscv_vfmacc_vf_f32m1(a7, r7[k], bv, vl);
        }
        __riscv_vse32_v_f32m1(C + (size_t)(i0 + 0) * ldc + jp + j, a0, vl);
        __riscv_vse32_v_f32m1(C + (size_t)(i0 + 1) * ldc + jp + j, a1, vl);
        __riscv_vse32_v_f32m1(C + (size_t)(i0 + 2) * ldc + jp + j, a2, vl);
        __riscv_vse32_v_f32m1(C + (size_t)(i0 + 3) * ldc + jp + j, a3, vl);
        __riscv_vse32_v_f32m1(C + (size_t)(i0 + 4) * ldc + jp + j, a4, vl);
        __riscv_vse32_v_f32m1(C + (size_t)(i0 + 5) * ldc + jp + j, a5, vl);
        __riscv_vse32_v_f32m1(C + (size_t)(i0 + 6) * ldc + jp + j, a6, vl);
        __riscv_vse32_v_f32m1(C + (size_t)(i0 + 7) * ldc + jp + j, a7, vl);
        j += (int)vl;
      }
    }
  }
}

/* V1: MR=4 with e32m4 (32 lanes) -- 4 accumulators + the B vector fit. */
static void gemm_m4_mr4(int M, int N, int K, const float *A, int lda,
                        const float *B, int ldb, float *C, int ldc) {
  const int panel_n = 256, MR = 4;
  for (int jp = 0; jp < N; jp += panel_n) {
    const int jn = (N - jp < panel_n) ? (N - jp) : panel_n;
#pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 <= M - MR; i0 += MR) {
      for (int j = 0; j < jn;) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(jn - j));
        const float *b = B + jp + j;
        vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0.f, vl);
        vfloat32m4_t a1 = __riscv_vfmv_v_f_f32m4(0.f, vl);
        vfloat32m4_t a2 = __riscv_vfmv_v_f_f32m4(0.f, vl);
        vfloat32m4_t a3 = __riscv_vfmv_v_f_f32m4(0.f, vl);
        const float *r0 = A + (size_t)(i0 + 0) * lda;
        const float *r1 = A + (size_t)(i0 + 1) * lda;
        const float *r2 = A + (size_t)(i0 + 2) * lda;
        const float *r3 = A + (size_t)(i0 + 3) * lda;
        for (int k = 0; k < K; ++k) {
          vfloat32m4_t bv = __riscv_vle32_v_f32m4(b + (size_t)k * ldb, vl);
          a0 = __riscv_vfmacc_vf_f32m4(a0, r0[k], bv, vl);
          a1 = __riscv_vfmacc_vf_f32m4(a1, r1[k], bv, vl);
          a2 = __riscv_vfmacc_vf_f32m4(a2, r2[k], bv, vl);
          a3 = __riscv_vfmacc_vf_f32m4(a3, r3[k], bv, vl);
        }
        __riscv_vse32_v_f32m4(C + (size_t)(i0 + 0) * ldc + jp + j, a0, vl);
        __riscv_vse32_v_f32m4(C + (size_t)(i0 + 1) * ldc + jp + j, a1, vl);
        __riscv_vse32_v_f32m4(C + (size_t)(i0 + 2) * ldc + jp + j, a2, vl);
        __riscv_vse32_v_f32m4(C + (size_t)(i0 + 3) * ldc + jp + j, a3, vl);
        j += (int)vl;
      }
    }
  }
}

/* V2: MR=8 with e32m2 (16 lanes). */
static void gemm_m2_mr8(int M, int N, int K, const float *A, int lda,
                        const float *B, int ldb, float *C, int ldc) {
  const int panel_n = 256, MR = 8;
  for (int jp = 0; jp < N; jp += panel_n) {
    const int jn = (N - jp < panel_n) ? (N - jp) : panel_n;
#pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 <= M - MR; i0 += MR) {
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

typedef void (*gemm_fn)(int, int, int, const float *, int, const float *, int,
                        float *, int);

static double bench(gemm_fn f, int M, int N, int K, const float *A,
                    const float *B, float *C, int rep) {
  f(M, N, K, A, 512, B, 512, C, 512); /* warm */
  double t0 = now();
  for (int r = 0; r < rep; ++r) f(M, N, K, A, N, B, N, C, N);
  return (now() - t0) / rep;
}

int main(void) {
  const int M = 512, N = 512, K = 512;
  float *A = (float *)aligned_alloc(64, (size_t)M * K * 4);
  float *B = (float *)aligned_alloc(64, (size_t)K * N * 4);
  float *C = (float *)aligned_alloc(64, (size_t)M * N * 4);
  float *R = (float *)aligned_alloc(64, (size_t)M * N * 4);
  for (int i = 0; i < M * K; ++i) A[i] = 0.001f * (i % 13);
  for (int i = 0; i < K * N; ++i) B[i] = 0.001f * (i % 17);

  gemm_m1_mr8(M, N, K, A, K, B, N, R, N); /* reference */

  const char *names[3] = {"m1/MR8 (current)", "m4/MR4", "m2/MR8"};
  gemm_fn fns[3] = {gemm_m1_mr8, gemm_m4_mr4, gemm_m2_mr8};
  for (int v = 0; v < 3; ++v) {
    memset(C, 0, (size_t)M * N * 4);
    double dt = bench(fns[v], M, N, K, A, B, C, 5);
    double err = 0;
    for (int i = 0; i < M * N; ++i) {
      double d = fabs(C[i] - R[i]);
      if (d > err) err = d;
    }
    printf("%-16s %7.1f ms  %6.1f GFLOP/s  max|err|=%.3g\n", names[v],
           dt * 1e3, 2.0 * M * N * K / dt / 1e9, err);
  }
  return 0;
}
