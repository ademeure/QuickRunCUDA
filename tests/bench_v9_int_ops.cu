// V9: Integer op variants — IADD, IMAD, ISHL, XOR peak
#ifndef OP
#define OP 0  // 0=IADD, 1=IMAD, 2=ISHL, 3=XOR, 4=IAND
#endif
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
#if OP == 0
                asm volatile("add.s32 %0, %0, %1;" : "+r"(v[k]) : "r"(b[k]));
#elif OP == 1
                asm volatile("mad.lo.s32 %0, %0, %1, %0;" : "+r"(v[k]) : "r"(b[k]));
#elif OP == 2
                asm volatile("shl.b32 %0, %0, 1;" : "+r"(v[k]));
#elif OP == 3
                asm volatile("xor.b32 %0, %0, %1;" : "+r"(v[k]) : "r"(b[k]));
#elif OP == 4
                asm volatile("and.b32 %0, %0, %1;" : "+r"(v[k]) : "r"(b[k]));
#endif
            }
        }
    }

    int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc += v[k];
    if (acc == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
