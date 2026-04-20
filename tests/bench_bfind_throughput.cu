// bfind throughput — find leading set bit (= floor(log2)).
// Mode 0: __ffs (find first set, low-order)
// Mode 1: __clz (count leading zeros, related to bfind)
// Mode 2: PTX bfind.u32
// Mode 3: PTX bfind.shiftamt.u32 (returns shift amount instead of position)
// Mode 4: PTX popc (for comparison)

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++)
        v[k] = (unsigned)(threadIdx.x * 131 + k * 17) | 0xF;  // ensure non-zero

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) {
#if MODE == 0
            v[k] ^= __ffs(v[k]);
#elif MODE == 1
            v[k] ^= __clz(v[k]);
#elif MODE == 2
            unsigned int b;
            asm volatile("bfind.u32 %0, %1;" : "=r"(b) : "r"(v[k]));
            v[k] ^= b;
#elif MODE == 3
            unsigned int b;
            asm volatile("bfind.shiftamt.u32 %0, %1;" : "=r"(b) : "r"(v[k]));
            v[k] ^= b;
#elif MODE == 4
            v[k] ^= __popc(v[k]);
#endif
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
