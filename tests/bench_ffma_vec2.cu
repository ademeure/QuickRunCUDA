// Vec2 FP32 FFMA via PTX `fma.rn.f32x2` (sm_100+, Blackwell).
// Question: can we escape the ~128 SASS-inst/SM/clk dispatch ceiling by
// packing 2 FP32 FMAs into a single warp-inst? If the SASS emits a single
// FFMA2 (or equivalent 64-bit packed FMA), then at ~128/SM/clk we'd see
// 256 FP32 FMAs/SM/clk — twice nominal FFMA.

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Each chain holds a packed float2 (64-bit) in a single .b64 reg.
    unsigned long long v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        float lo = 1.0001f + 0.0001f * (threadIdx.x + k * 23);
        float hi = 1.0002f + 0.0001f * (threadIdx.x + k * 29);
        unsigned int ulo = __float_as_int(lo);
        unsigned int uhi = __float_as_int(hi);
        v[k] = ((unsigned long long)uhi << 32) | ulo;
    }

    // Constants: packed (1.000001, 1.000001) and (0.9999, 0.9999)
    unsigned int c1_u = __float_as_int(1.000001f);
    unsigned int c0_u = __float_as_int(0.9999f);
    unsigned long long c1 = ((unsigned long long)c1_u << 32) | c1_u;
    unsigned long long c0 = ((unsigned long long)c0_u << 32) | c0_u;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;"
                    : "+l"(v[k]) : "l"(c1), "l"(c0));
            }
        }
    }

    unsigned long long acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if (acc == (unsigned long long)seed) ((unsigned long long*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
