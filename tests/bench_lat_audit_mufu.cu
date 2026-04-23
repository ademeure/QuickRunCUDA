// Audit: MUFU latency via single-thread serial chain (clock64 bracketed)
#ifndef CHAIN_LEN
#define CHAIN_LEN 4096
#endif
#ifndef OP
#define OP 0  // 0=ex2 1=rsq 2=rcp 3=sqrt 4=sin 5=cos 6=lg2 7=tanh 8=rcp.rn 9=sqrt.rn
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    float f = (float)(seed + 1) * 0.001f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) {
#if OP == 0
        asm volatile("ex2.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 1
        asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 2
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 3
        asm volatile("sqrt.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 4
        asm volatile("sin.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 5
        asm volatile("cos.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 6
        asm volatile("lg2.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 7
        asm volatile("tanh.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 8
        asm volatile("rcp.rn.f32 %0, %0;" : "+f"(f));
#elif OP == 9
        asm volatile("sqrt.rn.f32 %0, %0;" : "+f"(f));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((float*)C)[2] = f;
    }
}
