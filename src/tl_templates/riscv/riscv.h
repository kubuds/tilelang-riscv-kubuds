/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tl_templates/riscv/riscv.h
 * \brief Entry header for generated RISC-V kernels.
 *
 * Force-included into every translation unit of a generated RISC-V kernel
 * (tilelang/riscv/toolchain.py).  It provides the generic TileLang C
 * templates plus the RVV primitives the RISC-V lowering calls.
 *
 * The primitives are written against the RVV 1.0 intrinsic API and do not
 * assume a concrete VLEN: the vector length comes from vsetvl/vsetvlmax, so
 * the same source is correct on the K3 X100 (VLEN=256) and A100 (VLEN=1024)
 * clusters.
 *
 * The exported names are `tl_riscv_*` with C linkage because TileLang's C
 * codegen emits its own prototype for every `T.call_extern` symbol.  Keeping
 * the declarations in this header (force-included before the generated
 * prototypes) makes both declarations agree exactly; the wrappers below
 * therefore spell out `int32_t`/`float*` instead of nicer C++ types.
 */
#ifndef TL_TEMPLATES_RISCV_RISCV_H_
#define TL_TEMPLATES_RISCV_RISCV_H_

#include "tl_templates/cpp/common.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace tl {
namespace rv {

/* ------------------------------------------------------------------ *
 * fp32 GEMM: C[M,N] = A[M,K] * B[K,N], row-major, explicit strides.
 *
 * Two things matter on X100 and both were measured, not guessed:
 *
 *  * N is split into panels so a B panel stays in L2 while all M rows
 *    stream past it.
 *  * the k loop runs over MR=8 rows at once.  A single row leaves only one
 *    dependent vfmacc chain; vfmacc latency (~8-10 cycles) then dominates and
 *    the kernel measures 4.4 GFLOP/s per core.  Eight independent chains
 *    turn the same loop into an issue-bound one.
 *  * the column vector is e32m2 (16 lanes at VLEN=256), not e32m1.  With m1 a
 *    256-wide B panel is swept 32 times per k step and the 2 KB-strided loads
 *    dominate; m2 halves that.  Measured through bench_riscv_gemm.py on X100:
 *    m1 5.5 GFLOP/s, m2 30.1-31.3 GFLOP/s at 4 threads.  Eight threads collapse
 *    the panel streaming to 5.9 GFLOP/s, so 4 is the thread count to use for
 *    this kernel (the memory-bound GEMV path is the opposite: 7).
 *
 * beta semantics follow BLAS: beta==0 overwrites C (first k tile),
 * beta==1 accumulates (subsequent tiles).
 * ------------------------------------------------------------------ */
static inline void gemm_f32_impl(int M, int N, int K, const float *A,
                                 int lda, const float *B, int ldb, float *C,
                                 int ldc, int beta) {
  if (M <= 0 || N <= 0 || K <= 0) {
    return;
  }
  const int panel_n = 256;
  const int MR = 8;

  for (int jp = 0; jp < N; jp += panel_n) {
    const int jn = (N - jp < panel_n) ? (N - jp) : panel_n;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i0 = 0; i0 < M; i0 += MR) {
      if (i0 + MR <= M) {
        for (int j = 0; j < jn;) {
          size_t vl = __riscv_vsetvl_e32m2((size_t)(jn - j));
          const float *b = B + (size_t)jp + j;
          vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
          vfloat32m2_t a1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
          vfloat32m2_t a2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
          vfloat32m2_t a3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
          vfloat32m2_t a4 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
          vfloat32m2_t a5 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
          vfloat32m2_t a6 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
          vfloat32m2_t a7 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
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
          float *c0 = C + (size_t)(i0 + 0) * ldc + jp + j;
          float *c1 = C + (size_t)(i0 + 1) * ldc + jp + j;
          float *c2 = C + (size_t)(i0 + 2) * ldc + jp + j;
          float *c3 = C + (size_t)(i0 + 3) * ldc + jp + j;
          float *c4 = C + (size_t)(i0 + 4) * ldc + jp + j;
          float *c5 = C + (size_t)(i0 + 5) * ldc + jp + j;
          float *c6 = C + (size_t)(i0 + 6) * ldc + jp + j;
          float *c7 = C + (size_t)(i0 + 7) * ldc + jp + j;
          if (beta != 0) {
            a0 = __riscv_vfadd_vv_f32m2(a0, __riscv_vle32_v_f32m2(c0, vl), vl);
            a1 = __riscv_vfadd_vv_f32m2(a1, __riscv_vle32_v_f32m2(c1, vl), vl);
            a2 = __riscv_vfadd_vv_f32m2(a2, __riscv_vle32_v_f32m2(c2, vl), vl);
            a3 = __riscv_vfadd_vv_f32m2(a3, __riscv_vle32_v_f32m2(c3, vl), vl);
            a4 = __riscv_vfadd_vv_f32m2(a4, __riscv_vle32_v_f32m2(c4, vl), vl);
            a5 = __riscv_vfadd_vv_f32m2(a5, __riscv_vle32_v_f32m2(c5, vl), vl);
            a6 = __riscv_vfadd_vv_f32m2(a6, __riscv_vle32_v_f32m2(c6, vl), vl);
            a7 = __riscv_vfadd_vv_f32m2(a7, __riscv_vle32_v_f32m2(c7, vl), vl);
          }
          __riscv_vse32_v_f32m2(c0, a0, vl);
          __riscv_vse32_v_f32m2(c1, a1, vl);
          __riscv_vse32_v_f32m2(c2, a2, vl);
          __riscv_vse32_v_f32m2(c3, a3, vl);
          __riscv_vse32_v_f32m2(c4, a4, vl);
          __riscv_vse32_v_f32m2(c5, a5, vl);
          __riscv_vse32_v_f32m2(c6, a6, vl);
          __riscv_vse32_v_f32m2(c7, a7, vl);
          j += (int)vl;
        }
      } else {
        for (int i = i0; i < M; ++i) {
          const float *a = A + (size_t)i * lda;
          float *c = C + (size_t)i * ldc + jp;
          int j = 0;
          while (j < jn) {
            size_t vl = __riscv_vsetvl_e32m1((size_t)(jn - j));
            vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (int k = 0; k < K; ++k) {
              vfloat32m1_t bv =
                  __riscv_vle32_v_f32m1(B + (size_t)jp + j + (size_t)k * ldb, vl);
              acc = __riscv_vfmacc_vf_f32m1(acc, a[k], bv, vl);
            }
            if (beta != 0) {
              acc = __riscv_vfadd_vv_f32m1(
                  acc, __riscv_vle32_v_f32m1(c + j, vl), vl);
            }
            __riscv_vse32_v_f32m1(c + j, acc, vl);
            j += (int)vl;
          }
        }
      }
    }
  }
}

} // namespace rv
} // namespace tl

extern "C" {

/* Prototype must match what TileLang generates for T.call_extern:
 * int32_t(int32_t, int32_t, int32_t, float*, int32_t, float*, int32_t,
 *         float*, int32_t, int32_t). */
static inline int32_t tl_riscv_gemm_f32(int32_t M, int32_t N, int32_t K,
                                        float *A, int32_t lda, float *B,
                                        int32_t ldb, float *C, int32_t ldc,
                                        int32_t beta) {
  tl::rv::gemm_f32_impl(M, N, K, A, lda, B, ldb, C, ldc, beta);
  return 0;
}

} // extern "C"

/* The LLM primitives (GEMV, RMSNorm, RoPE, attention) are header-only and
 * only reachable through T.call_extern from the tilelang/riscv/llm driver;
 * pulling them in here keeps the toolchain to a single -include. */
#include "tl_templates/riscv/llm.h"
#include "tl_templates/riscv/ime2.h"

#endif // TL_TEMPLATES_RISCV_RISCV_H_
