/* M-row Q8 GEMV: one weight read serves 8 activation rows.  This is the
 * prefill primitive -- decode stays at M=1. */
#ifndef TL_RV_GEMV_M8
#define TL_RV_GEMV_M8

#include <stdint.h>
#include <riscv_vector.h>

static inline float tl_hsum4(vfloat32m4_t v, size_t vl) {
  vfloat32m1_t s = __riscv_vfredusum_vs_f32m4_f32m1(
      v, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
  return __riscv_vfmv_f_s_f32m1_f32(s);
}

static inline float tl_h2f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, f;
  if (exp == 0) {
    if (!man) f = sign;
    else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400u)) { man <<= 1; exp--; }
      man &= 0x3ffu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7f800000u | (man << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float o;
  __builtin_memcpy(&o, &f, 4);
  return o;
}

/* x: [8][K] row-major, y: [8][N] row-major */
static inline void gemv_q8_m8_impl(int N, int K, const uint8_t *q,
                                   const uint16_t *s, const float *x, float *y) {
  const int kb = K >> 5;

#pragma omp parallel for schedule(static)
  for (int n = 0; n < N; ++n) {
    const int8_t *qr = (const int8_t *)q + (size_t)n * K;
    const uint16_t *sr = s + (size_t)n * kb;
    size_t vl32 = __riscv_vsetvl_e32m4((size_t)32);
    vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0.0f, vl32);
    vfloat32m4_t a1 = __riscv_vfmv_v_f_f32m4(0.0f, vl32);
    vfloat32m4_t a2 = __riscv_vfmv_v_f_f32m4(0.0f, vl32);
    vfloat32m4_t a3 = __riscv_vfmv_v_f_f32m4(0.0f, vl32);
    vfloat32m4_t a4 = __riscv_vfmv_v_f_f32m4(0.0f, vl32);
    vfloat32m4_t a5 = __riscv_vfmv_v_f_f32m4(0.0f, vl32);
    vfloat32m4_t a6 = __riscv_vfmv_v_f_f32m4(0.0f, vl32);
    vfloat32m4_t a7 = __riscv_vfmv_v_f_f32m4(0.0f, vl32);
    for (int b = 0; b < kb; ++b) {
      const int off = b << 5;
      size_t vl = __riscv_vsetvl_e8m1((size_t)32);
      vint8m1_t wv = __riscv_vle8_v_i8m1(qr + off, vl);
      vfloat32m4_t wf =
          __riscv_vfcvt_f_x_v_f32m4(__riscv_vsext_vf4_i32m4(wv, vl), vl);
      wf = __riscv_vfmul_vf_f32m4(wf, tl_h2f(sr[b]), vl);
      a0 = __riscv_vfmacc_vv_f32m4(a0, wf, __riscv_vle32_v_f32m4(x + off, vl), vl);
      a1 = __riscv_vfmacc_vv_f32m4(a1, wf, __riscv_vle32_v_f32m4(x + K + off, vl), vl);
      a2 = __riscv_vfmacc_vv_f32m4(a2, wf, __riscv_vle32_v_f32m4(x + 2 * K + off, vl), vl);
      a3 = __riscv_vfmacc_vv_f32m4(a3, wf, __riscv_vle32_v_f32m4(x + 3 * K + off, vl), vl);
      a4 = __riscv_vfmacc_vv_f32m4(a4, wf, __riscv_vle32_v_f32m4(x + 4 * K + off, vl), vl);
      a5 = __riscv_vfmacc_vv_f32m4(a5, wf, __riscv_vle32_v_f32m4(x + 5 * K + off, vl), vl);
      a6 = __riscv_vfmacc_vv_f32m4(a6, wf, __riscv_vle32_v_f32m4(x + 6 * K + off, vl), vl);
      a7 = __riscv_vfmacc_vv_f32m4(a7, wf, __riscv_vle32_v_f32m4(x + 7 * K + off, vl), vl);
    }
    y[(size_t)0 * N + n] = tl_hsum4(a0, vl32);
    y[(size_t)1 * N + n] = tl_hsum4(a1, vl32);
    y[(size_t)2 * N + n] = tl_hsum4(a2, vl32);
    y[(size_t)3 * N + n] = tl_hsum4(a3, vl32);
    y[(size_t)4 * N + n] = tl_hsum4(a4, vl32);
    y[(size_t)5 * N + n] = tl_hsum4(a5, vl32);
    y[(size_t)6 * N + n] = tl_hsum4(a6, vl32);
    y[(size_t)7 * N + n] = tl_hsum4(a7, vl32);
  }
}

#endif
