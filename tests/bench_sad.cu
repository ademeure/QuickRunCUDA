// sad.s32 (sum-absolute-difference) throughput
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int v = (int)threadIdx.x + (int)u2;
    int b = (int)threadIdx.x * 2 + (int)u2;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // baseline (just IADD)
        v = v + b;
#elif MODE == 1
        // sad.s32 (signed)
        asm("sad.s32 %0, %0, %1, %2;" : "+r"(v) : "r"(b), "r"(i));
#elif MODE == 2
        // sad.u32 (unsigned)
        asm("sad.u32 %0, %0, %1, %2;" : "+r"(v) : "r"(b), "r"(i));
#elif MODE == 3
        // __sad CUDA intrinsic
        v = __sad(v, b, (int)i);
#elif MODE == 4
        // Manual abs(a-b) + c
        v = abs(v - b) + (int)i;
#endif
    }

    if (v == seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = (unsigned)v;
}
