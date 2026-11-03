/* smt.vmadot.hp correctness, doc 5.4.7 vector.
 * Note: vsetvli must write a real register on this core (rd=x0 traps). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <fcntl.h>

static int8_t A[8][16] __attribute__((aligned(64)));
static int8_t B[8][16] __attribute__((aligned(64)));
static _Float16 S[64] __attribute__((aligned(64)));
static _Float16 C[64] __attribute__((aligned(64)));

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int fd = open("/proc/set_ai_thread", O_WRONLY);
  if (fd >= 0) { char b[32]; int n = snprintf(b, sizeof b, "%d", getpid()); ssize_t w = write(fd, b, n); (void)w; close(fd); }
  for (int r = 0; r < 8; ++r)
    for (int k = 0; k < 16; ++k) A[r][k] = (int8_t)(r + k);
  for (int c = 0; c < 8; ++c)
    for (int k = 0; k < 16; ++k) B[c][k] = (int8_t)(c + k);
  for (int i = 0; i < 64; ++i) S[i] = (float)(i % 8 + 1);
  memset(C, 0, sizeof(C));

  asm volatile(
      "vsetvli t0, zero, e32, m1, ta, ma\n\t"
      "vmv.v.i v16, 0\n\t"
      "vsetvli t0, zero, e16, m1, ta, ma\n\t"
      "vle16.v v0, (%3)\n\t"
      "vsetvli t0, zero, e8, m1, ta, ma\n\t"
      "vle8.v  v2, (%0)\n\t"
      "vle8.v  v8, (%1)\n\t"
      "vmadot.hp v16, v2, v8, v0, 0, i8\n\t"
      "vsetvli t0, zero, e16, m1, ta, ma\n\t"
      "vse16.v v16, (%2)\n\t"
      :: "r"(A), "r"(B), "r"(C), "r"(S)
      : "t0", "v0", "v2", "v8", "v16", "memory");

  int bad = 0;
  for (int r = 0; r < 8; ++r)
    for (int c = 0; c < 8; ++c) {
      int acc = 0;
      for (int k = 0; k < 16; ++k) acc += (r + k) * (c + k);
      float want = (float)acc * (float)(c + 1);
      float got = (float)C[r * 8 + c];
      if (got != want) {
        if (bad < 4) printf("  [%d][%d] got %.1f want %.1f\n", r, c, got, want);
        bad++;
      }
    }
  printf("vmadot.hp i8: %s (%d/64 mismatch)\n", bad ? "FAIL" : "PASS", bad);
  printf("row0:");
  for (int c = 0; c < 8; ++c) printf(" %.0f", (float)C[c]);
  printf("\n");
  return 0;
}
