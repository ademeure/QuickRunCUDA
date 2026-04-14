// Prevent identity DCE by alternating two ops that don't compose to identity.

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
    float f = 0.3141592f;
    unsigned long long total_dt = 0;

    #pragma unroll 1
    for (int outer = 0; outer < ITERS_OUTER; outer++) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll
        for (int j = 0; j < N_OPS; j++) {
#if OP == 0
            // rcp; rsqrt; rcp; rsqrt... breaks rcp(rcp(x))=x
            asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(f));
            asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 1
            // tanh; ex2; tanh; ex2
            asm volatile("tanh.approx.ftz.f32 %0, %0;" : "+f"(f));
            asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 2
            // Reference: ex2; rsqrt (should be sum of individual latencies)
            asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f));
            asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 3
            // PRMT chain via bit-rotating with DIFFERENT control each iter
            unsigned int u = __float_as_int(f);
            asm volatile("prmt.b32 %0, %0, %0, 0x3021;" : "+r"(u));  // non-identity perm
            f = __int_as_float(u);
#elif OP == 4
            // LOP3 self-chain with MAJ function (non-associative, non-identity)
            unsigned int u = __float_as_int(f);
            asm volatile("lop3.b32 %0, %0, %0, %0, 0x2A;" : "+r"(u));  // a different LUT
            f = __int_as_float(u);
#elif OP == 5
            // IMAD self with constant multiplier (non-identity)
            unsigned int u = __float_as_int(f);
            asm volatile("mad.lo.u32 %0, %0, 3, 1;" : "+r"(u));
            f = __int_as_float(u);
#elif OP == 6
            // IADD3 chain
            unsigned int u = __float_as_int(f);
            asm volatile("add.u32 %0, %0, %1;" : "+r"(u) : "r"(0x12345u));
            f = __int_as_float(u);
#elif OP == 7
            // SHF left-rotate by 1
            unsigned int u = __float_as_int(f);
            asm volatile("shf.l.wrap.b32 %0, %0, %0, 1;" : "+r"(u));
            f = __int_as_float(u);
#endif
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total_dt += (t1 - t0);
    }
    if (__float_as_int(f) == seed) ((float*)C)[0] = f;
    ((unsigned long long*)C)[1] = total_dt;
}
