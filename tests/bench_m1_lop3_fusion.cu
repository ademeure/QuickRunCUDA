// M1: LOP3 fusion — how many ops can compiler collapse into one LOP3?
// LOP3 takes 3 inputs, applies arbitrary 3-input boolean function (256 truth tables)
// MODE 0: a & b & c — single LOP3 (1 op)
// MODE 1: a & b & c | d — 4 inputs, can't fit in one LOP3
// MODE 2: (a & b) ^ (c & d) — 2 LOP3 needed
// MODE 3: a | (b & c) — 1 LOP3 (3 inputs)
// MODE 4: a ^ b ^ c — 1 LOP3
// MODE 5: a + b (XOR-style integer add via LOP3?) — no, IADD3
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int a = (unsigned)(threadIdx.x ^ u2);
    unsigned int b = 0xDEADBEEFu ^ (unsigned)u2;
    unsigned int c = 0xCAFEBABEu;
    unsigned int d = 0x12345678u;
    unsigned int v = a;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            v = v & b & c;
#elif MODE == 1
            v = (v & b & c) | d;
#elif MODE == 2
            v = (v & b) ^ (c & d);
#elif MODE == 3
            v = v | (b & c);
#elif MODE == 4
            v = v ^ b ^ c;
#elif MODE == 5
            v = (v & b) | (~v & c);  // multiplexer pattern
#elif MODE == 6
            v = v ^ b;  // 2-input XOR — should this be LOP3?
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/op=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS/16.0);
    }
}
