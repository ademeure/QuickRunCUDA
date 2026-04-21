// H2/H3/H4: pipe-specific power
// MODE 0: IMAD chain (Cluster A integer)
// MODE 1: LOP3 chain (Cluster B integer)
// MODE 2: MUFU rsqrt chain (XU pipe)
// MODE 3: MUFU sin/cos (slow XU)
// MODE 4: LDG.E chain (LSU)
// MODE 5: pure NOP-like (loop only)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)(threadIdx.x ^ u2);
    unsigned int x = (unsigned)threadIdx.x | 0xDEADBEEFu;
    float fa = (float)threadIdx.x * 0.001f + (float)u2 * 1e-9f;
    float fb = 1.0001f, fc = 0.5f;
    float m = (float)threadIdx.x * 0.5f + 1.0f;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            // IMAD (Cluster A integer)
            asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(v) : "r"(x), "r"(7u));
#elif MODE == 1
            // LOP3 (Cluster B integer)
            asm volatile("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(v) : "r"(x), "r"(7u));
#elif MODE == 2
            // MUFU rsqrt (XU pipe)
            asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
#elif MODE == 3
            // MUFU sin (also XU but slower)
            asm volatile("sin.approx.ftz.f32 %0, %0;" : "+f"(m));
#elif MODE == 4
            // LDG.E (LSU pipe) — read from A
            unsigned int z;
            asm volatile("ld.global.u32 %0, [%1];" : "=r"(z) : "l"(A + ((gtid + i + u) & 1023)));
            v ^= z;
#elif MODE == 5
            // baseline: empty (loop only)
            asm volatile("");
#endif
        }
    }

    if (v == (unsigned)seed && fa*m == 12345.6f) C[blockIdx.x] = (float)v + fa + m;
}
