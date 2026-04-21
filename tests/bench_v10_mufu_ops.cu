// V10: MUFU op variants — rsqrt/rcp/sin/cos/exp/log/sqrt latency
#ifndef OP
#define OP 0  // 0=rsqrt, 1=rcp, 2=sin, 3=cos, 4=ex2, 5=lg2, 6=sqrt
#endif
#ifndef CHAIN_LEN
#define CHAIN_LEN 1024
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;
    float v = (float)(seed + 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 16
    for (int i = 0; i < CHAIN_LEN; i++) {
#if OP == 0
        asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(v));
#elif OP == 1
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(v));
#elif OP == 2
        asm volatile("sin.approx.f32 %0, %0;" : "+f"(v));
#elif OP == 3
        asm volatile("cos.approx.f32 %0, %0;" : "+f"(v));
#elif OP == 4
        asm volatile("ex2.approx.f32 %0, %0;" : "+f"(v));
#elif OP == 5
        asm volatile("lg2.approx.f32 %0, %0;" : "+f"(v));
#elif OP == 6
        asm volatile("sqrt.approx.f32 %0, %0;" : "+f"(v));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    ((unsigned long long*)C)[0] = t1 - t0;
    ((float*)C)[2] = v;
}
