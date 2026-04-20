// Predicated branch via PTX vs CUDA if
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    unsigned int v = (unsigned)threadIdx.x + (unsigned)u2;
    unsigned int extra = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        v = v * 31u + (unsigned)i;
#if MODE == 0
        // No branch (baseline)
        extra++;
#elif MODE == 1
        // CUDA if statement (always true)
        if ((unsigned)i + (unsigned)u2 != 0xFFFFFFFFu) extra++;
#elif MODE == 2
        // PTX @p bra (always taken via predicate)
        asm volatile("{ .reg .pred p; setp.ne.b32 p, %0, 0xFFFFFFFF; @p bra ALWAYS_$0; ALWAYS_$0: }"
                     :: "r"(v));
        extra++;
#elif MODE == 3
        // PTX bra.uni (uniform branch — guaranteed convergent)
        asm volatile("bra.uni LBL_$0; LBL_$0:");
        extra++;
#elif MODE == 4
        // PTX @p exit early-return pattern
        asm volatile("{ .reg .pred p; setp.eq.b32 p, %0, 0xDEADBEEF; @p ret; }"
                     :: "r"(v));
        extra++;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v + extra;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n", MODE, t1 - t0, (double)(t1-t0)/(double)ITERS);
    }
}
