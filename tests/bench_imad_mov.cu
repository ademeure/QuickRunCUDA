// IMAD.MOV throughput: nvcc emits this frequently as MOV alternative.
// Mode 0: MOV (PTX mov.u32)
// Mode 1: IMAD.MOV via mul-by-1-add-zero (mad.lo b, 1, 0) — implicit MOV
// Mode 2: chain via XOR with 0
// Mode 3: chain via OR with 0

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_CHAINS], b[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k] = (unsigned)(threadIdx.x * 131 + k * 17 + (unsigned)u2);
        b[k] = (unsigned)(threadIdx.x * 271 + k * 23);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) {
#if MODE == 0
            // mov chain: v[k] = b[k]; b[k] = v[k];  (alternating)
            asm volatile("mov.u32 %0, %1;" : "=r"(v[k]) : "r"(b[k]));
            asm volatile("mov.u32 %0, %1;" : "=r"(b[k]) : "r"(v[k]));
#elif MODE == 1
            // IMAD.MOV emulation via mad.lo.u32 with mul=1, add=0
            asm volatile("mad.lo.u32 %0, %1, 1, 0;" : "=r"(v[k]) : "r"(b[k]));
            asm volatile("mad.lo.u32 %0, %1, 1, 0;" : "=r"(b[k]) : "r"(v[k]));
#elif MODE == 2
            // XOR with 0 for chain (also a move-equivalent)
            asm volatile("xor.b32 %0, %1, 0;" : "=r"(v[k]) : "r"(b[k]));
            asm volatile("xor.b32 %0, %1, 0;" : "=r"(b[k]) : "r"(v[k]));
#elif MODE == 3
            // OR with 0
            asm volatile("or.b32 %0, %1, 0;" : "=r"(v[k]) : "r"(b[k]));
            asm volatile("or.b32 %0, %1, 0;" : "=r"(b[k]) : "r"(v[k]));
#endif
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
