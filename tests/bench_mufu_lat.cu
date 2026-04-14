// MUFU latency sweep: chained self-op, 1 warp, 1 chain.

#ifndef UNROLL
#define UNROLL 64
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float f = 0.5f + 0.0001f * threadIdx.x;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // ex2.approx
            asm volatile("ex2.approx.f32 %0, %0;" : "+f"(f));
            f = f * 0.5f + 0.25f;
#elif OP == 1  // rsqrt.approx
            f = f * 0.5f + 1.0f;
            asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 2  // rcp.approx
            f = f * 0.5f + 1.0f;
            asm volatile("rcp.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 3  // sqrt.approx
            f = f * 0.5f + 1.0f;
            asm volatile("sqrt.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 4  // sin.approx
            f = f * 0.001f;
            asm volatile("sin.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 5  // cos.approx
            f = f * 0.001f;
            asm volatile("cos.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 6  // lg2.approx
            f = f * 0.5f + 1.0f;
            asm volatile("lg2.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 7  // tanh.approx
            f = f * 0.001f;
            asm volatile("tanh.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 8  // rcp.rn.f32 (precise, not MUFU)
            f = f * 0.5f + 1.0f;
            asm volatile("rcp.rn.f32 %0, %0;" : "+f"(f));
#elif OP == 9  // sqrt.rn.f32 (precise)
            f = f * 0.5f + 1.0f;
            asm volatile("sqrt.rn.f32 %0, %0;" : "+f"(f));
#endif
        }
    }
    if (__float_as_int(f) == seed) ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = f;
}
