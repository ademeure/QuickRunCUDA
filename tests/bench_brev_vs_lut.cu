// __brevll (64-bit bit reverse) vs LUT-based reverse.
// Mode 0: __brevll intrinsic
// Mode 1: __brev (32-bit) intrinsic
// Mode 2: PTX brev.b32
// Mode 3: 8-bit-LUT software reverse (4 byte lookups + bit reverse via LUT)
// Mode 4: shift-based bit reverse algorithm

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned long long lv[N_CHAINS];
    unsigned int v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        lv[k] = (unsigned long long)threadIdx.x * 0x9E3779B97F4A7C15ull + k;
        v[k] = (unsigned)(threadIdx.x * 131 + k * 17);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) {
#if MODE == 0
            lv[k] = __brevll(lv[k]);
#elif MODE == 1
            v[k] = __brev(v[k]);
#elif MODE == 2
            asm volatile("brev.b32 %0, %0;" : "+r"(v[k]));
#elif MODE == 3
            // 8-bit LUT reverse for 32-bit
            static const unsigned char lut[256] = {
              #define R2(n) n, n+2*64, n+1*64, n+3*64
              #define R4(n) R2(n), R2(n+2*16), R2(n+1*16), R2(n+3*16)
              #define R6(n) R4(n), R4(n+2*4), R4(n+1*4), R4(n+3*4)
              R6(0), R6(2), R6(1), R6(3)
            };
            unsigned int x = v[k];
            v[k] = ((unsigned)lut[x & 0xFF] << 24)
                 | ((unsigned)lut[(x >> 8) & 0xFF] << 16)
                 | ((unsigned)lut[(x >> 16) & 0xFF] << 8)
                 | (unsigned)lut[x >> 24];
#elif MODE == 4
            // Shift-based bit reverse (no LUT, no intrinsic)
            unsigned int x = v[k];
            x = ((x & 0x55555555u) << 1) | ((x & 0xAAAAAAAAu) >> 1);
            x = ((x & 0x33333333u) << 2) | ((x & 0xCCCCCCCCu) >> 2);
            x = ((x & 0x0F0F0F0Fu) << 4) | ((x & 0xF0F0F0F0u) >> 4);
            x = ((x & 0x00FF00FFu) << 8) | ((x & 0xFF00FF00u) >> 8);
            x = (x << 16) | (x >> 16);
            v[k] = x;
#endif
        }
    }

    unsigned long long lacc = 0; unsigned int vacc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) { lacc ^= lv[k]; vacc ^= v[k]; }
    if ((int)lacc == seed && (int)vacc == seed)
        ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = (unsigned)lacc + vacc;
}
