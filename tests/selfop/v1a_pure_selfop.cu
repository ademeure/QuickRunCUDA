// Test 1a: Pure self-op FFMA, single chain
// fma.rn.f32 %0, %0, %0, %0  -- output reg = all 3 source regs
// One register Rd is used as src0/src1/src2 AND as destination
#ifndef N_INNER
#define N_INNER 1024
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (threadIdx.x >= 32) return;

    float v = (float)threadIdx.x * 0.001f + 1.0001f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
            // d = a*a + a   (all 3 PTX operands map to same vreg)
            asm volatile("fma.rn.f32 %0, %0, %0, %0;" : "+f"(v));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)v == seed) C[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
        printf("v1a_pure_selfop N_INNER=%d N_OUTER=%d total=%llu clk=%llu cy/op=%.4f\n",
               N_INNER, N_OUTER, total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
