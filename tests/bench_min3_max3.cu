// 3-input min/max (clamp idiom)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int v = (int)threadIdx.x + (int)u2;
    int b = 100;
    int c_lo = 0;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // 2-input max via PTX
        asm("max.s32 %0, %0, %1;" : "+r"(v) : "r"(b));
#elif MODE == 1
        // 3-input max via PTX
        asm("max.s32 %0, %0, %1; max.s32 %0, %0, %2;"
            : "+r"(v) : "r"(b), "r"((int)i));
#elif MODE == 2
        // PTX max3.s32 (single-inst 3-way max)
        asm("max.max.s32 %0, %0, %1, %2;" : "+r"(v) : "r"(b), "r"((int)i));
#elif MODE == 3
        // CUDA __vimax3_s32 (we tested earlier — SIMD)
        v = __vimax3_s32(v, b, (int)i);
#elif MODE == 4
        // Clamp: max(min(v, hi), lo) — common pattern
        v = max(min(v, b), c_lo);
#elif MODE == 5
        // PTX direct clamp via min/max
        asm("min.s32 %0, %0, %1; max.s32 %0, %0, %2;"
            : "+r"(v) : "r"(b), "r"(c_lo));
#endif
    }

    if (v == seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = (unsigned)v;
}
