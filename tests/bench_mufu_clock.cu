// Direct clock64 latency measurement for MUFU ops.
// Chain N back-to-back MUFUs bracketed by clock64 reads. Dep chain on single reg.
// No range reduction, no FFMA pollution.

#ifndef N_OPS
#define N_OPS 128   // chain length
#endif
#ifndef ITERS_OUTER
#define ITERS_OUTER 256
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Only lane 0 of block 0 measures
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Starting value — small random-ish non-zero
    float f = 0.3141592f;
    unsigned long long total_dt = 0;

    #pragma unroll 1
    for (int outer = 0; outer < ITERS_OUTER; outer++) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

        #pragma unroll
        for (int j = 0; j < N_OPS; j++) {
#if OP == 0
            asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 1
            asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 2
            asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 3
            asm volatile("sqrt.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 4
            asm volatile("sin.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 5
            asm volatile("cos.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 6
            asm volatile("lg2.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 7
            asm volatile("tanh.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 8
            // FFMA with 4 distinct sources to avoid read-port doubling: impossible with 1 reg
            // use a,a,b,c pattern where b,c are init-time constants
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f) : "f"(1.0000001f), "f"(1e-30f));
#elif OP == 9
            asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"(f) : "f"(1.0000001f));
#elif OP == 10
            unsigned int u = __float_as_int(f);
            asm volatile("prmt.b32 %0, %0, %0, 0x3210;" : "+r"(u));
            f = __int_as_float(u);
#elif OP == 11
            unsigned int u = __float_as_int(f);
            asm volatile("lop3.b32 %0, %0, %0, %0, 0xE8;" : "+r"(u));
            f = __int_as_float(u);
#elif OP == 12
            unsigned int u = __float_as_int(f);
            int r = (int)u;
            asm volatile("mad.lo.u32 %0, %0, %0, %0;" : "+r"(u));
            f = __int_as_float(u);
#endif
        }

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total_dt += (t1 - t0);
    }

    // Persist to avoid DCE and output total cycles
    if (__float_as_int(f) == seed) ((float*)C)[0] = f;
    ((unsigned long long*)C)[1] = total_dt;
}
