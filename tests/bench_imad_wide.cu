// IMAD wide multiply throughput/latency.
// Mode 0: regular IMAD (32x32 -> 32-bit, low part)
// Mode 1: wide multiply via mul.wide.u32 (32x32 -> 64-bit)
// Mode 2: high-half via mul.hi.u32
// Mode 3: __mul64hi (high part of 64x64 -> 128, top 64)
// Mode 4: __mulhi (32-bit signed high)

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_CHAINS], b[N_CHAINS];
    unsigned long long lv[N_CHAINS], lb[N_CHAINS];

    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k] = (unsigned)(threadIdx.x * 131 + k * 17 + (unsigned)u2);
        b[k] = (unsigned)(threadIdx.x * 271 + k * 23);
        lv[k] = (unsigned long long)v[k] * 0x9E3779B97F4A7C15ull;
        lb[k] = (unsigned long long)b[k] * 0x9E3779B97F4A7C15ull;
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) {
#if MODE == 0
            // Regular IMAD: 32x32 -> 32 (low)
            v[k] = v[k] * b[k] + b[k];
#elif MODE == 1
            // mul.wide.u32: 32x32 -> 64
            unsigned long long w;
            asm volatile("mul.wide.u32 %0, %1, %2;" : "=l"(w) : "r"(v[k]), "r"(b[k]));
            v[k] = (unsigned)(w >> 32) ^ (unsigned)w;
#elif MODE == 2
            // mul.hi.u32: 32x32 -> 32 (high)
            unsigned int hi;
            asm volatile("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(v[k]), "r"(b[k]));
            v[k] = hi;
#elif MODE == 3
            // 64x64 -> 64 (low) via mul.lo.u64
            lv[k] = lv[k] * lb[k];
#elif MODE == 4
            // 64x64 -> 64 (high)
            lv[k] = __umul64hi(lv[k], lb[k]);
#endif
        }
    }

    unsigned int acc = 0;
    unsigned long long lacc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) { acc ^= v[k]; lacc ^= lv[k]; }
    if (acc == (unsigned)seed && (int)lacc == seed)
        ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc + (unsigned)lacc;
}
