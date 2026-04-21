// mov.b32 register copy cost
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)threadIdx.x;
    unsigned int b = (unsigned)threadIdx.x * 2 + (unsigned)u2;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Just chain v through arithmetic (baseline)
        v = v ^ (b + (unsigned)i);
#elif MODE == 1
        // mov.b32 v from b, then arith
        asm("mov.b32 %0, %1;" : "=r"(v) : "r"(b + (unsigned)i));
        v = v ^ b;
#elif MODE == 2
        // explicit mov chain (v = b; b = v;)
        unsigned int t;
        asm("mov.b32 %0, %1;" : "=r"(t) : "r"(b));
        asm("mov.b32 %0, %1;" : "=r"(b) : "r"(v));
        asm("mov.b32 %0, %1;" : "=r"(v) : "r"(t));
        v += (unsigned)i;
#elif MODE == 3
        // OR with 0 (often used as MOV substitute)
        asm("or.b32 %0, %1, 0;" : "=r"(v) : "r"(b + (unsigned)i));
        v ^= b;
#endif
    }

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v + b;
}
