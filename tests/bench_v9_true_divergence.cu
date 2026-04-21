// V9: TRUE branch divergence with different instruction types
// Distinguishes predication (fast) from actual warp subset serialization.
#ifndef PATTERN
#define PATTERN 0
#endif
#ifndef CHAIN_LEN
#define CHAIN_LEN 1024
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    int lane = threadIdx.x & 31;
    float v = (float)(lane + 1) * 0.001f;
    float b = (float)(lane + 2) * 0.001f;

#if PATTERN == 0
    int group = 0;  // all uniform: just FFMA
#elif PATTERN == 1
    int group = lane & 1;  // 2-way: FFMA vs rsqrt
#elif PATTERN == 2
    int group = lane & 3;  // 4-way: FFMA, rsqrt, FADD, FMUL
#elif PATTERN == 3
    int group = lane & 7;  // 8-way: + sin, cos, exp, log
#endif

    unsigned long long t0, t1;
    if (lane == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    __syncwarp();

    // Each path uses a DIFFERENT instruction type — can't be predicated into one
    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
        if (group == 0) {
            v = v * b + v;  // FFMA
        } else if (group == 1) {
            asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(v));  // MUFU
        } else if (group == 2) {
            v = v + b;  // FADD
        } else if (group == 3) {
            v = v * b;  // FMUL
        } else if (group == 4) {
            asm volatile("sin.approx.f32 %0, %0;" : "+f"(v));  // MUFU.sin
        } else if (group == 5) {
            asm volatile("cos.approx.f32 %0, %0;" : "+f"(v));  // MUFU.cos
        } else if (group == 6) {
            asm volatile("ex2.approx.f32 %0, %0;" : "+f"(v));  // MUFU.ex2
        } else {
            asm volatile("lg2.approx.f32 %0, %0;" : "+f"(v));  // MUFU.lg2
        }
    }

    __syncwarp();
    if (lane == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
    }
    if (v == 1.234567e-30f) C[lane + 4] = v;
}
