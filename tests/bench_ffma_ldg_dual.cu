// FFMA (FMA pipe) + LDG (LSU pipe) dual-issue test.
// Expectation per Hopper/Blackwell docs: separate pipes, should overlap fully.
// Mode 0: FFMA only (NC=8 chains) — uses 2 unique sources to avoid 3rd-port limit
// Mode 1: LDG only (NC=8 independent loads, hot in L1)
// Mode 2: FFMA + LDG mixed per chain step

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
    float fv[N_CHAINS], fb[N_CHAINS];
    unsigned int lv[N_CHAINS];

    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        fv[k] = (float)(threadIdx.x + k);
        fb[k] = (float)(threadIdx.x * 2 + k);
        lv[k] = (unsigned)(threadIdx.x * 131 + k * 17);
    }

    // Set up small hot region in A so LDG hits L1
    unsigned int* ai = (unsigned int*)A;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if MIX_MODE == 0
                // 2-source FFMA (avoids RF port limit)
                asm volatile("fma.rn.f32 %0, %0, %1, %0;"
                             : "+f"(fv[k]) : "f"(fb[k]));
#elif MIX_MODE == 1
                // LDG via L1 cache (.ca = cache-all). Address from chain to defeat hoist.
                unsigned int x;
                unsigned int idx = (lv[k] & 0x3F) + k * 64;  // small footprint, hits L1
                asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(ai + idx));
                lv[k] = x ^ lv[k];
#elif MIX_MODE == 2
                // Both per chain step (LDG chain-dependent)
                asm volatile("fma.rn.f32 %0, %0, %1, %0;"
                             : "+f"(fv[k]) : "f"(fb[k]));
                unsigned int x;
                unsigned int idx = (lv[k] & 0x3F) + k * 64;
                asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(ai + idx));
                lv[k] = x ^ lv[k];
#elif MIX_MODE == 3
                // LDG only, NO chain dep (idx depends on i but result discarded)
                unsigned int x;
                unsigned int idx = (i + k * 64 + (unsigned)u2 * lv[k]) & 0xFF;
                asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(ai + idx));
                lv[k] = lv[k] + x;  // accumulate but doesn't gate next LDG
#elif MIX_MODE == 4
                // FFMA + LDG, LDG has NO chain dep on FFMA or own chain
                asm volatile("fma.rn.f32 %0, %0, %1, %0;"
                             : "+f"(fv[k]) : "f"(fb[k]));
                unsigned int x;
                unsigned int idx = (i + k * 64 + (unsigned)u2 * lv[k]) & 0xFF;
                asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(ai + idx));
                lv[k] = lv[k] + x;
#endif
            }
        }
    }

    float facc = 0.0f; unsigned int lacc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) { facc += fv[k]; lacc ^= lv[k]; }
    if (((int)facc == seed) && ((int)lacc == seed))
        ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = lacc;
}
