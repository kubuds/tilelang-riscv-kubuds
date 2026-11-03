/* Throughput of smt.vmadot.hp (int4/int8 tiles), compute only, A100. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

static int8_t W[512] __attribute__((aligned(64)));
static _Float16 S[64] __attribute__((aligned(64)));

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int fd = open("/proc/set_ai_thread", O_WRONLY);
  if (fd >= 0) {
    char b[32];
    int n = snprintf(b, sizeof b, "%d", getpid());
    if (write(fd, b, n) < 0) perror("write");
    close(fd);
  }
  for (int i = 0; i < 512; ++i) W[i] = (int8_t)((i * 7) % 61 - 30);
  for (int i = 0; i < 64; ++i) S[i] = 1.0f;

  const int iters = 50000;
  long n;
  double t0, dt;

  n = iters;
  t0 = now();
  asm volatile(
      "vsetvli t0, zero, e32, m1, ta, ma\n\t"
      "vmv.v.i v16, 0\n\t"
      "vsetvli t0, zero, e16, m1, ta, ma\n\t"
      "vle16.v v0, (%[s])\n\t"
      "vsetvli t0, zero, e8, m1, ta, ma\n\t"
      "vle8.v  v2, (%[a])\n\t"
      "vle8.v  v8, (%[b])\n\t"
      "1:\n\t"
      "smt.vmadot.hp v16, v2, v8, v0, 0, i8\n\t"
      "addi %[n], %[n], -1\n\t"
      "bnez %[n], 1b\n\t"
      : [n] "+r"(n)
      : [a] "r"(W), [b] "r"(W + 256), [s] "r"(S)
      : "t0", "v0", "v2", "v8", "v16", "memory");
  dt = now() - t0;
  printf("hp i8 : %.2f ms for %d ops -> %.1f GMAC/s\n", dt * 1e3, iters,
         (double)iters * 1024 / dt / 1e9);

  n = iters;
  t0 = now();
  asm volatile(
      "vsetvli t0, zero, e32, m1, ta, ma\n\t"
      "vmv.v.i v16, 0\n\t"
      "vsetvli t0, zero, e16, m1, ta, ma\n\t"
      "vle16.v v0, (%[s])\n\t"
      "vsetvli t0, zero, e8, m1, ta, ma\n\t"
      "vle8.v  v2, (%[a])\n\t"
      "vle8.v  v8, (%[b])\n\t"
      "1:\n\t"
      "smt.vmadot.hp v16, v2, v8, v0, 0, i4\n\t"
      "addi %[n], %[n], -1\n\t"
      "bnez %[n], 1b\n\t"
      : [n] "+r"(n)
      : [a] "r"(W), [b] "r"(W + 256), [s] "r"(S)
      : "t0", "v0", "v2", "v8", "v16", "memory");
  dt = now() - t0;
  printf("hp i4 : %.2f ms for %d ops -> %.1f GMAC/s\n", dt * 1e3, iters,
         (double)iters * 2048 / dt / 1e9);
  return 0;
}
