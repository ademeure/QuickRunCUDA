// HMMA + FFMA overlap: can FFMA hide HMMA's chain latency?
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int a0 = (threadIdx.x ^ u2) | 0x11111111u;
    unsigned int a1 = (threadIdx.x ^ u2) | 0x22222222u;
    unsigned int a2 = (threadIdx.x ^ u2) | 0x33333333u;
    unsigned int a3 = (threadIdx.x ^ u2) | 0x44444444u;
    unsigned int b0 = (threadIdx.x ^ u2) | 0x55555555u;
    unsigned int b1 = (threadIdx.x ^ u2) | 0x66666666u;
    float c0=0,c1=0,c2=0,c3=0;
    float x = (float)(threadIdx.x ^ u2) * 0.001f;
    float y = (float)(threadIdx.x ^ u2) * 0.002f + 1.0f;
    float z = (float)(threadIdx.x ^ u2) * 0.003f + 0.5f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        a0 ^= i;
#define HMMA \
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#define FFMA32 \
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z; \
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z; \
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z; \
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z; \
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z; \
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z; \
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z; \
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
#if MODE == 0
        // Pure HMMA (1 mma per iter)
        HMMA;
#elif MODE == 1
        // Pure FFMA (32 fmas per iter)
        FFMA32;
#elif MODE == 2
        // HMMA + 32 FFMA (interleaved) — if overlap, total < HMMA + FFMA
        HMMA;
        FFMA32;
#elif MODE == 3
        // HMMA + 16 FFMA
        HMMA;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
#elif MODE == 4
        // 4 HMMA (chained) — to compare with mode 5
        HMMA; HMMA; HMMA; HMMA;
#elif MODE == 5
        // 4 HMMA + 32 FFMA — full overlap test
        HMMA; HMMA; HMMA; HMMA;
        FFMA32;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sink = c0+c1+c2+c3+x;
    if ((int)sink == seed) C[blockIdx.x] = sink;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.2f\n", MODE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
