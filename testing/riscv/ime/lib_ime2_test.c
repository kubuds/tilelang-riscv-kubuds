#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>

void ime2_pack_b_only(int N, int K, const int8_t *B, int8_t *Bp);
void ime2_pack_a_only(int M, int K, const int8_t *A, int8_t *Ap);
void ime2_gemm_ldc(int M, int N, int K, const int8_t *Ap, const int8_t *Bp, int32_t *C, int ldc);

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int fd = open("/proc/set_ai_thread", O_WRONLY);
  if (fd >= 0) { char b[32]; int n = snprintf(b, sizeof b, "%d", getpid()); ssize_t w = write(fd, b, n); (void)w; close(fd); }
  { cpu_set_t cs; sched_getaffinity(0, sizeof(cs), &cs); printf("affinity:"); for (int i = 0; i < 16; i++) if (CPU_ISSET(i, &cs)) printf(" %d", i); printf("\n"); }

  const int M = 16, N = 32, K = 16;
  int8_t *A = calloc(M * K, 1), *B = calloc(K * N, 1);
  int8_t *Ap = calloc(M * K, 1), *Bp = calloc(N * K, 1);
  int32_t *C = calloc(M * N, 4);
  for (int i = 0; i < M * K; ++i) A[i] = (int8_t)(i % 17 - 8);
  for (int i = 0; i < K * N; ++i) B[i] = (int8_t)(i % 13 - 6);
  ime2_pack_b_only(N, K, B, Bp);
  ime2_pack_a_only(M, K, A, Ap);
  ime2_gemm_ldc(M, N, K, Ap, Bp, C, N);
  long long sum = 0;
  for (int i = 0; i < M * N; ++i) sum += C[i];
  printf("library gemm ran, checksum %lld, C[0]=%d\n", sum, C[0]);

  /* reference */
  long long ref0 = 0, refsum = 0;
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      int acc = 0;
      for (int k = 0; k < K; ++k) acc += A[m * K + k] * B[k * N + n];
      refsum += acc;
      if (m == 0 && n == 0) ref0 = acc;
    }
  }
  printf("reference checksum %lld, ref C[0]=%lld\n", refsum, ref0);
  return 0;
}
