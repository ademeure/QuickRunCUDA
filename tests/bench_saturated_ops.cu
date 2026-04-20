// Saturated arithmetic throughput on B300.
// Mode 0: add.s32 (no saturation, baseline)
// Mode 1: add.sat.s32 (saturation)
// Mode 2: sub.sat.s32
// Mode 3: cvt.sat.u8.s32 (saturating cvt)
// Mode 4: __sad (sum absolute diff)

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int v[N_CHAINS], b[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k] = (int)(threadIdx.x * 131 + k * 17 + (unsigned)u2);
        b[k] = (int)(threadIdx.x * 271 + k * 23);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) {
#if MODE == 0
            asm volatile("add.s32 %0, %0, %1;" : "+r"(v[k]) : "r"(b[k]));
#elif MODE == 1
            asm volatile("add.sat.s32 %0, %0, %1;" : "+r"(v[k]) : "r"(b[k]));
#elif MODE == 2
            asm volatile("sub.sat.s32 %0, %0, %1;" : "+r"(v[k]) : "r"(b[k]));
#elif MODE == 3
            // cvt.sat.u8.s32 (saturate to u8)
            asm volatile("cvt.sat.u8.s32 %0, %0;" : "+r"(v[k]));
            asm volatile("add.s32 %0, %0, %1;" : "+r"(v[k]) : "r"(b[k]));
#elif MODE == 4
            v[k] = __sad(v[k], b[k], 0);
#endif
        }
    }

    int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if (acc == seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = (unsigned)acc;
}
