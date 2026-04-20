// Multi-step carry chain (big-int add)
#ifndef N_WORDS
#define N_WORDS 4
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_WORDS], b[N_WORDS];
    #pragma unroll
    for (int k = 0; k < N_WORDS; k++) {
        v[k] = (unsigned)(threadIdx.x * 131 + k * 17 + (unsigned)u2);
        b[k] = (unsigned)(threadIdx.x * 271 + k * 23);
    }

    for (int i = 0; i < ITERS; i++) {
        // Big-int add: v += b with carry chain
#if N_WORDS == 2
        asm("add.cc.u32 %0, %0, %2; addc.u32 %1, %1, %3;"
            : "+r"(v[0]), "+r"(v[1]) : "r"(b[0]), "r"(b[1]));
#elif N_WORDS == 4
        asm("add.cc.u32 %0, %0, %4;"
            "addc.cc.u32 %1, %1, %5;"
            "addc.cc.u32 %2, %2, %6;"
            "addc.u32 %3, %3, %7;"
            : "+r"(v[0]), "+r"(v[1]), "+r"(v[2]), "+r"(v[3])
            : "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
#elif N_WORDS == 8
        asm("add.cc.u32 %0, %0, %8;"
            "addc.cc.u32 %1, %1, %9;"
            "addc.cc.u32 %2, %2, %10;"
            "addc.cc.u32 %3, %3, %11;"
            "addc.cc.u32 %4, %4, %12;"
            "addc.cc.u32 %5, %5, %13;"
            "addc.cc.u32 %6, %6, %14;"
            "addc.u32 %7, %7, %15;"
            : "+r"(v[0]), "+r"(v[1]), "+r"(v[2]), "+r"(v[3]),
              "+r"(v[4]), "+r"(v[5]), "+r"(v[6]), "+r"(v[7])
            : "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
              "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
#endif
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_WORDS; k++) acc ^= v[k];
    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
