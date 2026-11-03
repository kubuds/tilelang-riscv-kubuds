#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <fcntl.h>

static int8_t A[256] __attribute__((aligned(64)));
static int8_t B[256] __attribute__((aligned(64)));
static int32_t C32[64] __attribute__((aligned(64)));

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int fd = open("/proc/set_ai_thread", O_WRONLY);
  if (fd >= 0) { char b[32]; int n = snprintf(b, sizeof b, "%d", getpid()); ssize_t w = write(fd, b, n); (void)w; close(fd); }
  for (int i = 0; i < 256; ++i) { A[i] = (int8_t)(i % 17 - 8); B[i] = (int8_t)(i % 13 - 6); }
  memset(C32, 0, sizeof(C32));
  printf("step0 ok\n");

  asm volatile("vsetvli t0, zero, e32, m1, ta, ma\n\t"
               "vmv.v.i v16, 0\n\t"
               "vmv.v.i v17, 0\n\t" ::: "t0", "v16", "v17", "memory");
  printf("zero ok\n");

  asm volatile("vsetvli t0, zero, e8, m1, ta, ma\n\t"
               "vle8.v v4, (%0)\n\t"
               "vle8.v v8, (%1)\n\t" :: "r"(A), "r"(B) : "t0", "v4", "v8", "memory");
  printf("load ok\n");

  asm volatile("vsetvli t0, zero, e8, m1, ta, ma\n\t"
               "vmadot v16, v4, v8, i8\n\t" ::: "t0", "v16", "v17", "memory");
  printf("vmadot ok\n");

  asm volatile("vsetvli t0, zero, e32, m1, ta, ma\n\t"
               "vse32.v v16, (%0)\n\t" :: "r"(C32) : "t0", "memory");
  printf("store ok, C[0]=%d C[1]=%d\n", C32[0], C32[1]);
  return 0;
}
