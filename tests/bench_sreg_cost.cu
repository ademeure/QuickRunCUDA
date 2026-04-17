extern "C" __global__ __launch_bounds__(32, 1) void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    unsigned long long acc = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned long long v;
#if OP == 0
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(v));
#elif OP == 1
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
#elif OP == 2
            unsigned x;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(x));
            v = x;
#elif OP == 3
            unsigned x;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(x));
            v = x;
#elif OP == 4
            unsigned x;
            asm volatile("mov.u32 %0, %%nsmid;" : "=r"(x));
            v = x;
#elif OP == 5
            unsigned x;
            asm volatile("mov.u32 %0, %%laneid;" : "=r"(x));
            v = x;
#elif OP == 6
            unsigned x;
            asm volatile("mov.u32 %0, %%warpid;" : "=r"(x));
            v = x;
#endif
            acc ^= v + i + j;
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned long long*)C)[1] = acc;
    }
}
