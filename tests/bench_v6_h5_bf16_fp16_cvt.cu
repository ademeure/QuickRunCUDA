// V6 H5: BF16 ↔ FP16 cvt cost (rare path)
// MODE 0: BF16 → FP16 (via FP32 intermediate?)
// MODE 1: FP16 → BF16
// MODE 2: BF16 → FP32 (baseline)
// MODE 3: FP32 → BF16
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) v[j] = (unsigned int)((threadIdx.x + j + 1) * 1024);

    unsigned int out[8] = {0};

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned int xi = (unsigned int)i * 0x1234;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned int src = v[k * 2] ^ xi;
#if MODE == 0
            // BF16 → FP16 (via cvt.rn.f16.bf16 — exists?)
            asm volatile("cvt.rn.f16.bf16 %0, %1;" : "=r"(out[k]) : "r"(src));
#elif MODE == 1
            // FP16 → BF16
            asm volatile("cvt.rn.bf16.f16 %0, %1;" : "=r"(out[k]) : "r"(src));
#elif MODE == 2
            // BF16 → FP32
            float f;
            asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f) : "r"(src));
            out[k] = __float_as_uint(f);
#elif MODE == 3
            // FP32 → BF16
            float f = __uint_as_float(src);
            asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=r"(out[k]) : "f"(f));
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    unsigned int ssum = 0; for (int j = 0; j < 8; j++) ssum ^= out[j];
    if (threadIdx.x == 0 && blockIdx.x == 0) ((unsigned int*)C)[1] = ssum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d cy/iter=%.3f cy/cvt=%.3f\n",
               MODE, ITERS, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/8.0);
    }
}
