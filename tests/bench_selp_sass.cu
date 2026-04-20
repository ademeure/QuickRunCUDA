// selp PTX vs CUDA ternary - SASS comparison.

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int* p = (unsigned int*)A;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FF;
    unsigned int v = p[idx];
    unsigned int b = p[idx + 1];
    unsigned int result;

#if MODE == 0
    // CUDA ternary - compiler picks SASS form
    result = (v > b) ? v : b;
#elif MODE == 1
    // PTX selp directly
    asm("{ .reg .pred p; setp.gt.u32 p, %1, %2; selp.b32 %0, %1, %2, p; }"
        : "=r"(result) : "r"(v), "r"(b));
#elif MODE == 2
    // CUDA min via intrinsic
    result = max(v, b);
#elif MODE == 3
    // PTX max
    asm("max.u32 %0, %1, %2;" : "=r"(result) : "r"(v), "r"(b));
#endif

    p[idx] = result;
}
