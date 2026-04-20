// FFMA + ULDC parallel issue test.
// ULDC = uniform datapath constant load (fetches from constant cache to
// uniform register). Tests if FFMA can run in parallel with ULDC issues.

#ifndef MODE
#define MODE 0
#endif

extern "C" __constant__ unsigned int CMEM[1024];

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float fv = (float)threadIdx.x + 1.5f;
    float fb = 1.0000001f;
    float fc = 0.0000001f;
    unsigned int sum = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Pure FFMA chain (baseline)
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
#elif MODE == 1
        // Pure ULDC chain (constant memory load)
        unsigned int x;
        unsigned int idx = (i + (unsigned)u2 * sum) & 0x3FF;
        asm volatile("ld.const.u32 %0, [%1];" : "=r"(x) : "l"(CMEM + idx));
        sum ^= x;
#elif MODE == 2
        // FFMA + ULDC mixed
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
        unsigned int x;
        unsigned int idx = (i + (unsigned)u2 * sum) & 0x3FF;
        asm volatile("ld.const.u32 %0, [%1];" : "=r"(x) : "l"(CMEM + idx));
        sum ^= x;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)fv == seed && sum == (unsigned)seed) C[blockIdx.x] = fv + (float)sum;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
#if MODE == 2
        unsigned long long inst = (unsigned long long)ITERS * 2;
#else
        unsigned long long inst = (unsigned long long)ITERS;
#endif
        printf("MODE=%d insts=%llu clk=%llu cy/inst=%.3f cy/iter=%.3f\n",
               MODE, inst, t1 - t0,
               (double)(t1-t0)/(double)inst, (double)(t1-t0)/(double)ITERS);
    }
}
