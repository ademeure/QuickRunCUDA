// setp boolean composition variants
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)threadIdx.x + (unsigned)u2;
    unsigned int b = (unsigned)threadIdx.x * 2;

    for (int i = 0; i < ITERS; i++) {
        v = v * 31u + b;
#if MODE == 0
        // Single setp + use predicate
        unsigned int sel;
        asm("{ .reg .pred p; setp.gt.u32 p, %1, %2; selp.b32 %0, 1, 0, p; }"
            : "=r"(sel) : "r"(v), "r"(b));
        v += sel;
#elif MODE == 1
        // setp.AND fused (combine with prior pred)
        unsigned int sel;
        asm("{ .reg .pred p, q; setp.gt.u32 p, %1, 0; setp.lt.and.u32 q, %1, %2, p; selp.b32 %0, 1, 0, q; }"
            : "=r"(sel) : "r"(v), "r"(b));
        v += sel;
#elif MODE == 2
        // 2 separate setp + and instruction
        unsigned int sel;
        asm("{ .reg .pred p, q, r; setp.gt.u32 p, %1, 0; setp.lt.u32 q, %1, %2; and.pred r, p, q; selp.b32 %0, 1, 0, r; }"
            : "=r"(sel) : "r"(v), "r"(b));
        v += sel;
#elif MODE == 3
        // setp.OR fused
        unsigned int sel;
        asm("{ .reg .pred p, q; setp.gt.u32 p, %1, 0; setp.lt.or.u32 q, %1, %2, p; selp.b32 %0, 1, 0, q; }"
            : "=r"(sel) : "r"(v), "r"(b));
        v += sel;
#endif
    }

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
