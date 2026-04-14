// Approx vs precise throughput — what's the real cost?
// Uses explicit PTX so fast-math flag doesn't change emission.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef CHAINS
#define CHAINS 8
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float f[CHAINS];
    #pragma unroll
    for (int k=0;k<CHAINS;k++) f[k] = 1.2345f + 0.01f * k;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k=0;k<CHAINS;k++) {
#if OP == 0   // rsqrt.approx.ftz (fastest path)
                asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(f[k]));
#elif OP == 1 // rsqrt.approx.f32 (no ftz)
                asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(f[k]));
#elif OP == 2 // sqrt.approx.ftz
                asm volatile("sqrt.approx.ftz.f32 %0, %0;" : "+f"(f[k]));
#elif OP == 3 // sqrt.rn (precise — Newton-Raphson)
                asm volatile("sqrt.rn.f32 %0, %0;" : "+f"(f[k]));
#elif OP == 4 // rcp.approx.ftz
                asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(f[k]));
#elif OP == 5 // rcp.rn (precise)
                asm volatile("rcp.rn.f32 %0, %0;" : "+f"(f[k]));
#elif OP == 6 // div.approx.ftz
                asm volatile("div.approx.ftz.f32 %0, %0, %1;" : "+f"(f[k]) : "f"(1.7f));
#elif OP == 7 // div.full.ftz (default when not fast-math)
                asm volatile("div.full.ftz.f32 %0, %0, %1;" : "+f"(f[k]) : "f"(1.7f));
#elif OP == 8 // div.rn (precise)
                asm volatile("div.rn.f32 %0, %0, %1;" : "+f"(f[k]) : "f"(1.7f));
#elif OP == 9 // ex2.approx.ftz (baseline)
                asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f[k]));
#elif OP == 10 // FFMA (baseline for scalar FP)
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[k]) : "f"(1.0000001f), "f"(0.9999f));
#endif
            }
        }
    }
    float acc = 0;
    #pragma unroll
    for (int k=0;k<CHAINS;k++) acc += f[k];
    if (__float_as_int(acc) == seed) ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
