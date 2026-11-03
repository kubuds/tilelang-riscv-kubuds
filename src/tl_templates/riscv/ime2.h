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
 * \file tl_templates/riscv/ime2.h
 * \brief SpacemiT IME2 (smt.vmadot) int8 GEMM entry points.
 *
 * IME2 exists only on the K3 A100 cluster (cpu8-15, VLEN=1024) and executes
 * one vmadot as an 8x8x16 tile (1024 MAC).  The kernel consumes *pre-packed*
 * panels: A in 16-row x 16-k blocks, B in 32-column x 16-k blocks.  Packing is
 * a load-time cost by design; packing inside the k loop measured slower than
 * the GEMM itself (the same conclusion SpacemiT's own Triton tutorial reports),
 * so this backend keeps the two phases separate instead of hiding the pack in
 * a tile lowering.
 *
 * Usage from TIR (see testing/riscv/test_riscv_ime2.py):
 *   pack_b(N, K, B, Bp)        once, at weight load
 *   pack_a(M, K, A, Ap)        per activation batch
 *   gemm(M, N, K, Ap, Bp, C, ldc)
 *
 * Requirements: M%16==0, N%32==0, K%16==0, ldc>=N.  The kernel returns
 * without writing C when they are violated, so callers must pad.
 *
 * The implementation lives in ime2/ime2_gemm_k3.c (built into
 * libtl_riscv_ime2.so) because it needs -march=..._xsmtvdotii, an extension the
 * rest of TileLang must not inherit.
 */
#ifndef TL_TEMPLATES_RISCV_IME2_H_
#define TL_TEMPLATES_RISCV_IME2_H_

#include <stdint.h>

#if defined(TL_RISCV_IME2)

extern "C" {
void ime2_pack_a_only(int M, int K, const int8_t *A, int8_t *Ap);
void ime2_pack_b_only(int N, int K, const int8_t *B, int8_t *Bp);
void ime2_gemm_ldc(int M, int N, int K, const int8_t *Ap, const int8_t *Bp,
                   int32_t *C, int ldc);
}

/* Thin wrappers whose signatures match what TileLang's C codegen emits for
 * T.call_extern (int32 scalars, uint8/int32 pointers from the raw buffers). */
extern "C" {

static inline int32_t tl_riscv_ime2_pack_a(int32_t M, int32_t K, uint8_t *A,
                                           uint8_t *Ap) {
  ime2_pack_a_only(M, K, (const int8_t *)A, (int8_t *)Ap);
  return 0;
}

static inline int32_t tl_riscv_ime2_pack_b(int32_t N, int32_t K, uint8_t *B,
                                           uint8_t *Bp) {
  ime2_pack_b_only(N, K, (const int8_t *)B, (int8_t *)Bp);
  return 0;
}

static inline int32_t tl_riscv_ime2_gemm(int32_t M, int32_t N, int32_t K,
                                         uint8_t *Ap, uint8_t *Bp, int32_t *C,
                                         int32_t ldc) {
  ime2_gemm_ldc(M, N, K, (const int8_t *)Ap, (const int8_t *)Bp, C, ldc);
  return 0;
}

/* Bytes the packers need. */
static inline int32_t tl_riscv_ime2_pack_b_bytes(int32_t N, int32_t K) {
  return (N / 32) * (K / 16) * 512;
}
static inline int32_t tl_riscv_ime2_pack_a_bytes(int32_t M, int32_t K) {
  return (M / 16) * (K / 16) * 256;
}

} // extern "C"

#endif // TL_RISCV_IME2

#endif // TL_TEMPLATES_RISCV_IME2_H_
