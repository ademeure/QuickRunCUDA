// V6 J2: Predicate evaluation cost
// MODE 0: 8× FFMA (no predicate)
// MODE 1: 8× FFMA @P0 (all-true: P0 set so FFMA executes)
// MODE 2: 8× FFMA @P0 (all-false: P0 unset, FFMA skipped)
//
// Predict: MODE 1 = MODE 0 (same instruction, just predicate true)
//          MODE 2 < MODE 0 (issued but skipped)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float a = (float)(threadIdx.x + 1) * 0.001f;
    float b = (float)(threadIdx.x + 2) * 0.001f;
    float f0=a, f1=a, f2=a, f3=a, f4=a, f5=a, f6=a, f7=a;
    float k0=b*0.99f, k1=b*0.98f, k2=b*0.97f, k3=b*0.96f;
    float k4=b*0.95f, k5=b*0.94f, k6=b*0.93f, k7=b*0.92f;

    // Set predicate P0 based on MODE: must NOT be compile-time constant
    // for the predicate emission to happen
    int pflag = (seed > 0) ? 1 : 0;  // depends on runtime arg
#if MODE == 1
    pflag = 1;  // override to true
#elif MODE == 2
    pflag = 0;  // override to false
#endif

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // 8 FFMA (no predicate)
        f0 = f0 * k0 + b;
        f1 = f1 * k1 + b;
        f2 = f2 * k2 + b;
        f3 = f3 * k3 + b;
        f4 = f4 * k4 + b;
        f5 = f5 * k5 + b;
        f6 = f6 * k6 + b;
        f7 = f7 * k7 + b;
#else
        // 8 FFMA @P0 (predicated; pflag controls)
        asm volatile("{ .reg .pred p;\n"
                     "  setp.ne.s32 p, %2, 0;\n"
                     "  @p fma.rn.f32 %0, %0, %1, %3; }"
                     : "+f"(f0) : "f"(k0), "r"(pflag), "f"(b));
        asm volatile("{ .reg .pred p;\n"
                     "  setp.ne.s32 p, %2, 0;\n"
                     "  @p fma.rn.f32 %0, %0, %1, %3; }"
                     : "+f"(f1) : "f"(k1), "r"(pflag), "f"(b));
        asm volatile("{ .reg .pred p;\n"
                     "  setp.ne.s32 p, %2, 0;\n"
                     "  @p fma.rn.f32 %0, %0, %1, %3; }"
                     : "+f"(f2) : "f"(k2), "r"(pflag), "f"(b));
        asm volatile("{ .reg .pred p;\n"
                     "  setp.ne.s32 p, %2, 0;\n"
                     "  @p fma.rn.f32 %0, %0, %1, %3; }"
                     : "+f"(f3) : "f"(k3), "r"(pflag), "f"(b));
        asm volatile("{ .reg .pred p;\n"
                     "  setp.ne.s32 p, %2, 0;\n"
                     "  @p fma.rn.f32 %0, %0, %1, %3; }"
                     : "+f"(f4) : "f"(k4), "r"(pflag), "f"(b));
        asm volatile("{ .reg .pred p;\n"
                     "  setp.ne.s32 p, %2, 0;\n"
                     "  @p fma.rn.f32 %0, %0, %1, %3; }"
                     : "+f"(f5) : "f"(k5), "r"(pflag), "f"(b));
        asm volatile("{ .reg .pred p;\n"
                     "  setp.ne.s32 p, %2, 0;\n"
                     "  @p fma.rn.f32 %0, %0, %1, %3; }"
                     : "+f"(f6) : "f"(k6), "r"(pflag), "f"(b));
        asm volatile("{ .reg .pred p;\n"
                     "  setp.ne.s32 p, %2, 0;\n"
                     "  @p fma.rn.f32 %0, %0, %1, %3; }"
                     : "+f"(f7) : "f"(k7), "r"(pflag), "f"(b));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float fsum = f0+f1+f2+f3+f4+f5+f6+f7;
    if (fsum == 1.234567e-30f) C[blockIdx.x] = fsum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d cy/iter=%.3f\n", MODE, ITERS, (double)(t1-t0)/(double)ITERS);
    }
}
