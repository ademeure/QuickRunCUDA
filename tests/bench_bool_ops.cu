// Boolean op throughput in chain
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)threadIdx.x + (unsigned)u2;
    unsigned int b = 0xDEADBEEFu;
    unsigned int c = 0xC0FFEE00u;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // xor chain
        asm("xor.b32 %0, %0, %1;" : "+r"(v) : "r"(b));
#elif MODE == 1
        // and chain
        asm("and.b32 %0, %0, %1;" : "+r"(v) : "r"(b));
#elif MODE == 2
        // or chain
        asm("or.b32 %0, %0, %1;" : "+r"(v) : "r"(b));
#elif MODE == 3
        // not (single source)
        asm("not.b32 %0, %0;" : "+r"(v));
#elif MODE == 4
        // lop3 with all 3 sources (xor3)
        asm("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v) : "r"(b), "r"(c));
#elif MODE == 5
        // 3 ops: xor + and + or in chain
        asm("xor.b32 %0, %0, %1; and.b32 %0, %0, %2; or.b32 %0, %0, %1;"
            : "+r"(v) : "r"(b), "r"(c));
#endif
    }

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
