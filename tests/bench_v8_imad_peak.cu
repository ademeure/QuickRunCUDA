// V8: IMAD peak verification
// Theoretical: integer pipe same rate as FP32 on Blackwell → 76.97 TIOPS at 2032 MHz
// 2-source IMAD chain: d = a * b + a (reuses a)
#ifndef N_CHAINS
#define N_CHAINS 8
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(int* A, int* B, int* C, int ITERS, int seed, int u2) {
    int v[N_CHAINS], b[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k] = threadIdx.x + k;
        b[k] = threadIdx.x * 2 + k;
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                // IMAD.IADD: d = a * b + a (2-source like FFMA pattern)
                asm volatile("mad.lo.s32 %0, %0, %1, %0;" : "+r"(v[k]) : "r"(b[k]));
            }
        }
    }

    int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc += v[k];
    if (acc == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
