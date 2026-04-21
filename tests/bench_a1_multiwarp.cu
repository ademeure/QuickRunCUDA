// A1 follow-up: multi-warp residency on same SMSP
// 1 SM = 4 SMSPs. With launch_bounds(W*32, 1), W warps share 4 SMSPs.
// W=8 → 2 warps per SMSP. W=16 → 4 warps per SMSP.
// MODE 0 = pure FFMA, MODE 1 = pure LOP3, MODE 2 = mixed (interleaved across warps)
#ifndef MODE
#define MODE 0
#endif
#ifndef NWARPS
#define NWARPS 8
#endif

extern "C" __global__ __launch_bounds__(NWARPS*32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int wid = threadIdx.x / 32;

    float a0=(float)threadIdx.x*1.001f+(float)u2*1e-9f, a1=a0+0.1f;
    float ya=(float)(threadIdx.x^u2)*0.001f+1.0f, za=0.5f;
    unsigned int b0=(unsigned)threadIdx.x^(unsigned)u2, b1=b0+1;
    unsigned int yb=0xDEADBEEFu;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            // All warps do FFMA
            a0 = a0*ya + za; a1 = a1*ya + za;
#elif MODE == 1
            // All warps do LOP3
            asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b0) : "r"(yb), "r"(b1));
            asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b1) : "r"(yb), "r"(b0));
#elif MODE == 2
            // Half warps FFMA, half LOP3 (residency-level mixing)
            if ((wid & 1) == 0) {
                a0 = a0*ya + za; a1 = a1*ya + za;
            } else {
                asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b0) : "r"(yb), "r"(b1));
                asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b1) : "r"(yb), "r"(b0));
            }
#elif MODE == 3
            // All warps do BOTH FFMA and LOP3
            a0 = a0*ya + za; a1 = a1*ya + za;
            asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b0) : "r"(yb), "r"(b1));
            asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b1) : "r"(yb), "r"(b0));
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sink = a0+a1+(float)(b0^b1);
    if ((int)sink == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = sink;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d NWARPS=%d clk=%llu cy/iter=%.3f\n",
               MODE, NWARPS, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
