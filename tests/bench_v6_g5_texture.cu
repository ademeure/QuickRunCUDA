// V6 G5: Texture/RO cache vs regular L1 LDG
// MODE 0: regular load (LDG.E)
// MODE 1: __ldg() (cache.global.read_only path → LDG.E.NC.SYS or similar)
// MODE 2: PTX ld.global.nc directly
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float acc = 0.0f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int idx = (threadIdx.x + k * 32 + i) & 1023;
#if MODE == 0
            float v = A[idx];
#elif MODE == 1
            float v = __ldg(A + idx);
#elif MODE == 2
            float v;
            asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(A + idx));
#endif
            acc += v;
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (acc == 1.234567e-30f) C[blockIdx.x] = acc;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d cy/iter=%.3f cy/op=%.3f\n",
               MODE, ITERS, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/8.0);
    }
}
