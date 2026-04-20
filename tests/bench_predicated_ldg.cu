// Predicated LDG: does setp + @p ld combine into one SASS inst?

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int* p = (int*)A;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FF;
    int x = (int)idx;

#if MODE == 0
    // Simple unconditional load
    x ^= p[idx];
#elif MODE == 1
    // Conditional load via if statement
    if (idx > 0) x ^= p[idx];
#elif MODE == 2
    // PTX explicit setp + @p ld
    asm volatile("{ .reg .pred p; .reg .b32 t; setp.gt.s32 p, %1, 0; @p ld.global.u32 t, [%2]; @p xor.b32 %0, %0, t; }"
                 : "+r"(x) : "r"(idx), "l"(p + idx));
#elif MODE == 3
    // Compiler ternary: x = cond ? p[idx] : x
    x ^= (idx > 0) ? p[idx] : 0;
#endif

    p[idx] = x;
}
