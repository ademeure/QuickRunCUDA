// dp4a (4x INT8 dot product) throughput
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int v = (int)threadIdx.x + (int)u2;
    unsigned int a = 0x01020304u + (unsigned)u2;
    unsigned int b = 0x05060708u;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // dp4a (signed 4xINT8 dot)
        v = __dp4a(a, b, v);
#elif MODE == 1
        // PTX dp4a directly
        asm("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(v) : "r"(a), "r"(b));
#elif MODE == 2
        // dp2a (signed 2xINT16 dot)
        v = __dp2a_lo(a, b, v);
#elif MODE == 3
        // PTX dp2a.lo
        asm("dp2a.lo.s32.s32 %0, %1, %2, %0;" : "+r"(v) : "r"(a), "r"(b));
#elif MODE == 4
        // Manual 4-element add for comparison
        v = (int)((char)a * (char)b) + (int)((char)(a>>8) * (char)(b>>8))
          + (int)((char)(a>>16) * (char)(b>>16)) + (int)((char)(a>>24) * (char)(b>>24));
#endif
    }

    if (v == seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = (unsigned)v;
}
