// HFMA2 / BF16 FMA latency — clock64 bracketed.

#ifndef N_OPS
#define N_OPS 128
#endif
#ifndef ITERS_OUTER
#define ITERS_OUTER 256
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned int x = 0x3C003C10u;   // f16x2: two small positive half-floats
    unsigned short s = 0x3C10;
    unsigned long long total_dt = 0;

    #pragma unroll 1
    for (int outer = 0; outer < ITERS_OUTER; outer++) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll
        for (int j = 0; j < N_OPS; j++) {
#if OP == 0  // HFMA2 f16x2 self-op (x = x*x + x)
            asm volatile("fma.rn.f16x2 %0, %0, %0, %0;" : "+r"(x));
#elif OP == 1 // HFMA2.BF16 bf16x2
            asm volatile("fma.rn.bf16x2 %0, %0, %0, %0;" : "+r"(x));
#elif OP == 2 // HFMA2.RELU f16x2
            asm volatile("fma.rn.relu.f16x2 %0, %0, %0, %0;" : "+r"(x));
#elif OP == 3 // HADD2 f16x2
            asm volatile("add.rn.f16x2 %0, %0, %0;" : "+r"(x));
#elif OP == 4 // HMUL2 f16x2
            asm volatile("mul.rn.f16x2 %0, %0, %0;" : "+r"(x));
#elif OP == 5 // HADD2.BF16
            asm volatile("add.rn.bf16x2 %0, %0, %0;" : "+r"(x));
#elif OP == 6 // HMNMX2 min f16x2
            asm volatile("min.f16x2 %0, %0, %0;" : "+r"(x));
#elif OP == 7 // FFMA2 (FP32x2 vec)
            unsigned long long xx = ((unsigned long long)x << 32) | x;
            asm volatile("fma.rn.f32x2 %0, %0, %0, %0;" : "+l"(xx));
            x = (unsigned int)xx;
#elif OP == 8 // scalar HFMA
            asm volatile("fma.rn.f16 %0, %0, %0, %0;" : "+h"(s));
#elif OP == 9 // scalar HADD
            asm volatile("add.rn.f16 %0, %0, %0;" : "+h"(s));
#elif OP == 10 // HSETP2 + SELP
            unsigned int r;
            asm volatile("{.reg .pred p0,p1; setp.lt.f16x2 p0|p1, %1, %1; selp.b32 %0, %1, 0, p0;}" : "=r"(r) : "r"(x));
            x = r;
#endif
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total_dt += (t1 - t0);
    }
    if ((int)x == seed || (int)s == seed) ((unsigned int*)C)[0] = x + s;
    ((unsigned long long*)C)[1] = total_dt;
}
