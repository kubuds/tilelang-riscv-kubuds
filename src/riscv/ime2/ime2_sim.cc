/*
 * SpacemiT IME-2.0 matrix-multiply-accumulate (MMA) reference simulator.
 *
 * Purpose
 * -------
 * TileLang kernels target the SpacemiT IME-2.0 matrix engine of the K3 A100
 * AI-CPU through the TIR intrinsics in `tilelang/riscv/ime.py`, which lower to
 * the extern entry points implemented here:
 *
 *   spacemit_ime2_mma_i8    : D = A * B + C   A:(8x16 i8) B:(16x8 i8) C,D:(8x8 i32)
 *   spacemit_ime2_mma_f16   : D = A * B + C   A,B:(8x8 fp16)  C,D:(8x8 fp32)
 *   spacemit_ime2_mma_bf16  : D = A * B + C   A,B:(8x8 bf16)  C,D:(8x8 fp32)
 *
 * The shapes and data types follow the SpacemiT K3 paper ("SpacemiT K3: A
 * RVA23 RISC-V AI CPU with 60 TOPS AI Compute", Table 3, IME 2.0 row): the
 * 1024-bit shared MAC datapath performs one 8x16x8 INT8 or 8x8x8 FP16/BF16
 * matrix multiply-accumulate per MMA instruction, with operands and results
 * held in vector registers.
 *
 * The real IME-2.0 instruction encodings are proprietary and not part of the
 * public RISC-V ISA specifications, so this library defines the stable
 * software ABI between TileLang kernels and the engine.  On real silicon the
 * vendor runtime replaces this file with a driver/instruction-issuing
 * implementation of the same three entry points (the memory-operand contract
 * below is intentionally minimal so a register-resident fast path can be
 * substituted without changing kernel code).  For correctness testing -- and
 * as the executable specification of the ABI -- this simulator implements the
 * semantics in plain C, requiring only RV64GC.
 *
 * License: MIT (original code written for TileLang; shapes/types are facts
 * from the SpacemiT K3 paper, not copied code).
 */

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- FP16 / BF16 helpers (bit-exact conversions, no FPU requirements) ---- */

static float ime2_fp16_to_fp32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign; /* +/- 0 */
    } else {
      /* subnormal: normalize */
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp -= 1;
      }
      mant &= 0x3FFu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13); /* inf / nan */
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

static float ime2_bf16_to_fp32(uint16_t h) {
  uint32_t bits = (uint32_t)h << 16;
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

/* ---- IME-2.0 MMA entry points (see file comment for the contract) ---- */

/*
 * INT8 MMA: D(8x8) = A(8x16) * B(16x8) + C(8x8), row-major, i32 accumulate.
 * C and D may alias.
 */
void spacemit_ime2_mma_i8(const int8_t *A, const int8_t *B, const int32_t *C, int32_t *D) {
  int32_t acc[8][8];
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      int32_t sum = 0;
      for (int k = 0; k < 16; ++k) {
        sum += (int32_t)A[i * 16 + k] * (int32_t)B[k * 8 + j];
      }
      acc[i][j] = sum;
    }
  }
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      D[i * 8 + j] = C[i * 8 + j] + acc[i][j];
    }
  }
}

/*
 * FP16 MMA: D(8x8, fp32) = A(8x8, fp16) * B(8x8, fp16) + C(8x8, fp32).
 * C and D may alias.  Products accumulate in fp32 (the engine's fp32
 * accumulate path); the simulator uses double-rounding-free fp32 arithmetic
 * via the host FPU.
 */
void spacemit_ime2_mma_f16(const void *A, const void *B, const float *C, float *D) {
  const uint16_t *a = (const uint16_t *)A;
  const uint16_t *b = (const uint16_t *)B;
  float acc[8][8];
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < 8; ++k) {
        sum += ime2_fp16_to_fp32(a[i * 8 + k]) * ime2_fp16_to_fp32(b[k * 8 + j]);
      }
      acc[i][j] = sum;
    }
  }
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      D[i * 8 + j] = C[i * 8 + j] + acc[i][j];
    }
  }
}

/*
 * BF16 MMA: same contract as the FP16 variant with BF16 operands.
 */
void spacemit_ime2_mma_bf16(const void *A, const void *B, const float *C, float *D) {
  const uint16_t *a = (const uint16_t *)A;
  const uint16_t *b = (const uint16_t *)B;
  float acc[8][8];
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < 8; ++k) {
        sum += ime2_bf16_to_fp32(a[i * 8 + k]) * ime2_bf16_to_fp32(b[k * 8 + j]);
      }
      acc[i][j] = sum;
    }
  }
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      D[i * 8 + j] = C[i * 8 + j] + acc[i][j];
    }
  }
}

#ifdef __cplusplus
} /* extern "C" */
#endif
