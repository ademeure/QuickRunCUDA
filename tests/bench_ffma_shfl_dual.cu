// Definitive dispatch model test: FFMA (FMA pipe) + SHFL (LSU pipe).
// If pipes are independent, mixed time = max(FFMA, SHFL).
// If shared (unified-cluster theory), mixed time = sum.

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef MIX_MODE
#define MIX_MODE 2
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv[N_CHAINS];
    float fb[N_CHAINS];
    float fc[N_CHAINS];
    unsigned int sv[N_CHAINS];
    unsigned int sb[N_CHAINS];

    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        fv[k] = (float)(threadIdx.x + k);
        fb[k] = (float)(threadIdx.x * 2 + k);
        fc[k] = (float)(threadIdx.x * 3 + k);
        sv[k] = 0xDEAD0000u + (threadIdx.x * 131 + k * 17);
        sb[k] = 0xBEEF0000u + (threadIdx.x * 271 + k * 23);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if MIX_MODE == 0
                asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                             : "+f"(fv[k]) : "f"(fb[k]), "f"(fc[k]));
#elif MIX_MODE == 1
                asm volatile("shfl.sync.idx.b32 %0, %0, %1, 0x1f, 0xffffffff;"
                             : "+r"(sv[k]) : "r"(sb[k] & 0x1f));
#else
                // both: 1 FFMA + 1 SHFL per chain step, INDEPENDENT registers
                asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                             : "+f"(fv[k]) : "f"(fb[k]), "f"(fc[k]));
                asm volatile("shfl.sync.idx.b32 %0, %0, %1, 0x1f, 0xffffffff;"
                             : "+r"(sv[k]) : "r"(sb[k] & 0x1f));
#endif
            }
        }
    }

    float facc = 0.0f;
    unsigned int sacc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) { facc += fv[k]; sacc ^= sv[k]; }
    if (((int)sacc == seed) && ((int)facc == seed))
        ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = sacc;
}
