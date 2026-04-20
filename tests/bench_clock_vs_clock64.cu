// clock() vs clock64() vs %globaltimer cost
// Mode 0: clock() PTX inline (32-bit %clock)
// Mode 1: clock64() PTX inline (64-bit %clock64)
// Mode 2: %globaltimer (nanosecond clock)
// Mode 3: just NOP equivalent (loop with no clock read)

#ifndef N_READS
#define N_READS 8
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Strong anti-DCE: chain clock value into address used for next read
    unsigned long long acc = (unsigned long long)u2;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < N_READS; k++) {
#if MODE == 0
            unsigned int c;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
            acc = acc * 31u + c + (unsigned)i;
#elif MODE == 1
            unsigned long long c;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
            acc = acc * 31ull + c + (unsigned long long)i;
#elif MODE == 2
            unsigned long long c;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(c));
            acc = acc * 31ull + c + (unsigned long long)i;
#elif MODE == 3
            // No clock read, just ALU work matched to clock body
            acc = acc * 31ull + (unsigned long long)(i * 13 + k);
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)acc == seed) ((unsigned long long*)C)[blockIdx.x] = acc;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)ITERS * (unsigned long long)N_READS;
        printf("MODE=%d iters=%d N_READS=%d clk=%llu cy/op=%.3f\n",
               MODE, ITERS, N_READS, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
