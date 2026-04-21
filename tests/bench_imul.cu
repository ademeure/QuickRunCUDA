// integer multiply variants
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)threadIdx.x + (unsigned)u2;
    unsigned int b = 31u + (unsigned)u2;
    unsigned int c = 7u;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // mul.lo (multiply only, low 32 bits)
        asm("mul.lo.u32 %0, %0, %1;" : "+r"(v) : "r"(b));
#elif MODE == 1
        // mad.lo (multiply-add)
        asm("mad.lo.u32 %0, %0, %1, %2;" : "+r"(v) : "r"(b), "r"(c));
#elif MODE == 2
        // C-style v *= b
        v *= b;
#elif MODE == 3
        // mad.lo with constant multiplier
        asm("mad.lo.u32 %0, %0, 31, %1;" : "+r"(v) : "r"(c));
#elif MODE == 4
        // shift + add (replace multiply by 31 = (v<<5) - v)
        v = (v << 5) - v + c;
#endif
    }

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
