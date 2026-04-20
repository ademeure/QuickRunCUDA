// Performance: __ldg (LDG.E.CONSTANT) vs ld.global.ca (LDG.E.STRONG.SM).
// Same buffer, same stride, just different SASS variant.

#ifndef N_LOADS
#define N_LOADS 8
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int* ai = (unsigned int*)A;
    unsigned int v[N_LOADS];

    #pragma unroll
    for (int k = 0; k < N_LOADS; k++) v[k] = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < N_LOADS; k++) {
            // Different addresses each iteration to defeat hoisting
            unsigned int idx = ((unsigned)i * 256u + k * 256u + threadIdx.x + (unsigned)u2 * v[k]) & 0x3FFF;
#if MODE == 0
            v[k] ^= __ldg(ai + idx);
#elif MODE == 1
            unsigned int x;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(ai + idx));
            v[k] ^= x;
#elif MODE == 2
            unsigned int x;
            asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(ai + idx));
            v[k] ^= x;
#elif MODE == 3
            unsigned int x;
            asm volatile("ld.global.u32 %0, [%1];" : "=r"(x) : "l"(ai + idx));
            v[k] ^= x;
#endif
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_LOADS; k++) acc ^= v[k];
    if ((int)acc == seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
