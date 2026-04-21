// V9: Various op latencies via serial dependency chain
#ifndef CHAIN_LEN
#define CHAIN_LEN 4096
#endif
#ifndef OP
#define OP 0  // 0=FFMA, 1=FADD, 2=FMUL, 3=DFMA, 4=IMAD
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    float fa = (float)(seed + 1) * 0.001f;
    float fb = (float)(seed + 2) * 0.001f;
    double da = (double)(seed + 1) * 0.001;
    double db = (double)(seed + 2) * 0.001;
    int ia = seed + 1;
    int ib = seed + 2;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) {
#if OP == 0
        asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fa) : "f"(fb));
#elif OP == 1
        asm volatile("add.f32 %0, %0, %1;" : "+f"(fa) : "f"(fb));
#elif OP == 2
        asm volatile("mul.f32 %0, %0, %1;" : "+f"(fa) : "f"(fb));
#elif OP == 3
        asm volatile("fma.rn.f64 %0, %0, %1, %0;" : "+d"(da) : "d"(db));
#elif OP == 4
        asm volatile("mad.lo.s32 %0, %0, %1, %0;" : "+r"(ia) : "r"(ib));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    unsigned long long cycles = t1 - t0;
    if (blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = cycles;
        // Anti-DCE
        ((float*)C)[2] = fa + (float)da + (float)ia;
    }
}
