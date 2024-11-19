
extern "C" __global__ void kernel(const unsigned int *A, unsigned int *B, int sm, int unused1, int unused2) {
  unsigned int smid;
  asm("mov.u32 %0, %smid;" : "=r"(smid));

  if (threadIdx.x > 0) {
    return;
  }

  unsigned int value = 99999;
  unsigned int expected = smid;
  unsigned int desired = smid + 2;
  if ((smid % 2) == 0 && (smid <= 12 || smid == 110)) {
    if (smid == 110) {
      expected = 14;
    }
    while (true) {
      asm volatile ("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(value) : "l"(&B[0]), "r"(expected), "r"(expected));
      if (value == expected) {
        break;
      }
      __nanosleep(100000);
    }
  } else {
    return;
  }

  if(true) {
    unsigned int val = 0;

    constexpr int num_256B_iterations = 32768;
    constexpr int num_measurements = 250;
    constexpr bool discard_cache = false;

    if (smid == 0) {
      for (int i = 0; i < num_256B_iterations; i++) {
        unsigned long long total_duration = 0;
        const unsigned int *A_ptr = &A[i*64];
        // warmup cache
        val += A_ptr[val];
        asm volatile ("discard.global.L2 [%0], 128;" : : "l"(&A_ptr[val]));
        for (int k = 0; k < 25; k++) {
          unsigned long long start = clock64();

          unsigned int tmp;
          asm volatile ("ld.global.cg.u32 %0, [%1];" : "=r"(tmp) : "l"(&A_ptr[val]));
          val += tmp;

          unsigned long long end = clock64();
          asm volatile ("discard.global.L2 [%0], 128;" : : "l"(&A_ptr[val]));
          total_duration += end - start;
        }
        printf("%d, 999, %d\n", i, (total_duration / 25 > 500 ? 1000 : 0));
      }
    }

    for (int i = 0; i < num_256B_iterations; i++) {
      unsigned long long total_duration = 0;
      const unsigned int *A_ptr = &A[i*64];
      val += A_ptr[val]; // warmup cache
      for (int k = 0; k < num_measurements; k++) {
        unsigned long long start = clock64();

        unsigned int tmp;
        asm volatile ("ld.global.cg.u32 %0, [%1];" : "=r"(tmp) : "l"(&A_ptr[val]));
        val += tmp;

        unsigned long long end = clock64();
        if constexpr (discard_cache) {
          asm volatile ("discard.global.L2 [%0], 128;" : : "l"(&A_ptr[val]));
        }
        total_duration += end - start;
      }
      printf("%d, %d, %.3f\n", i, smid, (float)total_duration / (float)num_measurements);
    }
    if (val == 77) {
      B[threadIdx.x] = 3;
    }
  }

  asm volatile ("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(value) : "l"(&B[0]), "r"(expected), "r"(desired));
}
