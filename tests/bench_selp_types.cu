// selp typed variants
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)threadIdx.x;
    unsigned int b = (unsigned)threadIdx.x * 2 + (unsigned)u2;
    float fv = (float)v;

    for (int i = 0; i < ITERS; i++) {
        v = v * 31u + b;
#if MODE == 0
        // selp.b32
        unsigned int sel;
        asm("{ .reg .pred p; setp.gt.u32 p, %1, %2; selp.b32 %0, %1, %2, p; }"
            : "=r"(sel) : "r"(v), "r"(b));
        v = sel;
#elif MODE == 1
        // selp.u32
        unsigned int sel;
        asm("{ .reg .pred p; setp.gt.u32 p, %1, %2; selp.u32 %0, %1, %2, p; }"
            : "=r"(sel) : "r"(v), "r"(b));
        v = sel;
#elif MODE == 2
        // selp.s32
        int sel;
        asm("{ .reg .pred p; setp.gt.s32 p, %1, %2; selp.s32 %0, %1, %2, p; }"
            : "=r"(sel) : "r"((int)v), "r"((int)b));
        v = (unsigned)sel;
#elif MODE == 3
        // selp.f32
        float sel;
        asm("{ .reg .pred p; setp.gt.f32 p, %1, %2; selp.f32 %0, %1, %2, p; }"
            : "=f"(sel) : "f"(__uint_as_float(v)), "f"(__uint_as_float(b)));
        v = __float_as_uint(sel);
#endif
    }

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
