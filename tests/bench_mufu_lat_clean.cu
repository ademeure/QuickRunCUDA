// Clean MUFU latency — chained self-op, NO range-reduction FFMA.
// Relies on MUFU being non-idempotent so compiler cannot DCE a chain.

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
#if OP == 0  // ex2.approx.ftz — direct chain
            asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 1  // rsqrt.approx.ftz
            asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 2  // rcp.approx.ftz
            asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 3  // sqrt.approx.ftz
            asm volatile("sqrt.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 4  // sin.approx.ftz (idempotent-ish; sin(sin(x)) converges but not instantly)
            asm volatile("sin.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 5  // cos.approx.ftz
            asm volatile("cos.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 6  // lg2.approx.ftz
            asm volatile("lg2.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 7  // tanh.approx.ftz
            asm volatile("tanh.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 8  // FFMA (reference)
            asm volatile("fma.rn.f32 %0, %0, %0, %0;" : "+f"(f));
#elif OP == 9  // PRMT (reference for alu)
            unsigned int u = __float_as_int(f);
            asm volatile("prmt.b32 %0, %0, %0, 0x3210;" : "+r"(u));
            f = __int_as_float(u);
#endif
        }
    }
    if (__float_as_int(f) == seed) ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = f;
}
