// __builtin_assume impact on SASS — does telling the compiler save inst?

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int* p = (int*)A;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x);

#if MODE == 0
    // Baseline - bounds check
    if (idx < 1024) p[idx] = (int)u2;
#elif MODE == 1
    // Promise idx is always in bounds
    __builtin_assume(idx < 1024);
    if (idx < 1024) p[idx] = (int)u2;  // compiler should remove the check
#elif MODE == 2
    // Promise + no check
    __builtin_assume(idx >= 0 && idx < 1024);
    p[idx] = (int)u2;
#elif MODE == 3
    // Compute base + offset with __builtin_assume on alignment
    __builtin_assume(idx % 4 == 0);  // 4-aligned
    p[idx] = (int)u2;
#elif MODE == 4
    // Modulo with assumption
    int q = idx >> 5;
    __builtin_assume(q < 32);
    p[q & 31] = q;
#endif
}
