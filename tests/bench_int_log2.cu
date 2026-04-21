// int log2 via bfind vs CUDA logb / log2f
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)threadIdx.x + 1u + (unsigned)u2;
    int log_val = 0;

    for (int i = 0; i < ITERS; i++) {
        v = v * 31u + (unsigned)i + 1u;  // ensure non-zero
#if MODE == 0
        // bfind.shiftamt — direct bit-position to log2
        unsigned int b;
        asm("bfind.shiftamt.u32 %0, %1;" : "=r"(b) : "r"(v));
        log_val = (int)b;
#elif MODE == 1
        // bfind without shiftamt (returns position of MSB; need to add 1)
        unsigned int b;
        asm("bfind.u32 %0, %1;" : "=r"(b) : "r"(v));
        log_val = (int)b;
#elif MODE == 2
        // 31 - clz
        log_val = 31 - __clz((int)v);
#elif MODE == 3
        // log2f (FP path)
        log_val = (int)__log2f((float)v);
#elif MODE == 4
        // floorf(log2f) — proper int log2
        log_val = (int)floorf(__log2f((float)v));
#endif
    }

    if (log_val == seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = (unsigned)log_val;
}
