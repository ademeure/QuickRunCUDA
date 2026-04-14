// F2FP + wgmma/hmma co-issue (tensor core)
// Uses mma.sync.m16n8k16 f16 (or f16) for simplicity.
// If tensor core shares any pipe with F2FP, F2FP throughput should drop.
// Otherwise, time = max(F2FP, TC).

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef N_CVT
#define N_CVT 16
#endif
#ifndef N_MMA
#define N_MMA 0
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    unsigned int p[N_CVT];
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);

    // MMA accumulators (4× f32)
    float d0 = threadIdx.x * 0.01f;
    float d1 = (threadIdx.x + 1) * 0.01f;
    float d2 = (threadIdx.x + 2) * 0.01f;
    float d3 = (threadIdx.x + 3) * 0.01f;
    unsigned int a0 = 0x3C003C00u ^ threadIdx.x;
    unsigned int a1 = 0x3C003C01u ^ threadIdx.x;
    unsigned int a2 = 0x3C003C02u ^ threadIdx.x;
    unsigned int a3 = 0x3C003C03u ^ threadIdx.x;
    unsigned int b0 = 0x3C003C00u ^ (threadIdx.x + 7);
    unsigned int b1 = 0x3C003C01u ^ (threadIdx.x + 7);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CVT; k++) {
                asm volatile(
                    "{ .reg .b16 _h;\n\t"
                    "  cvt.rn.satfinite.e4m3x2.f16x2 _h, %0;\n\t"
                    "  cvt.rn.f16x2.e4m3x2 %0, _h; }"
                    : "+r"(p[k]));
            }
            #pragma unroll
            for (int m = 0; m < N_MMA; m++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n\t"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) acc ^= p[k];
    acc ^= __float_as_int(d0) ^ __float_as_int(d1) ^ __float_as_int(d2) ^ __float_as_int(d3);
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
