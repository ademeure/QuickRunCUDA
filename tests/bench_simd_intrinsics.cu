// Video SIMD intrinsics throughput.
// Mode 0: __viaddmax_s32 (signed add then max with 0)
// Mode 1: __vimax3_s32 (3-way max signed)
// Mode 2: __vsadu4 (sum of absolute differences, 4-byte)
// Mode 3: __vmaxs2 (2x16-bit signed max)
// Mode 4: __vmaxu4 (4x8-bit unsigned max)
// Mode 5: __vavgu2 (2x16-bit averaging)

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_CHAINS], b[N_CHAINS], c[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k] = (unsigned)(threadIdx.x * 131 + k * 17 + (unsigned)u2);
        b[k] = (unsigned)(threadIdx.x * 271 + k * 23);
        c[k] = (unsigned)(threadIdx.x * 419 + k * 41);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) {
#if MODE == 0
            v[k] = __viaddmax_s32(v[k], b[k], 0);
#elif MODE == 1
            v[k] = __vimax3_s32(v[k], b[k], c[k]);
#elif MODE == 2
            v[k] = __vsadu4(v[k], b[k]);
#elif MODE == 3
            v[k] = __vmaxs2(v[k], b[k]);
#elif MODE == 4
            v[k] = __vmaxu4(v[k], b[k]);
#elif MODE == 5
            v[k] = __vavgu2(v[k], b[k]);
#endif
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
