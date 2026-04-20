// PTX vshl/vshr (vector shift) - what SASS does it emit?
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int* p = (unsigned int*)A;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FF;
    unsigned int v = (unsigned)idx + (unsigned)u2;
    unsigned int amt = (unsigned)idx & 31;

#if MODE == 0
    // Standard left shift
    asm("shl.b32 %0, %0, %1;" : "+r"(v) : "r"(amt));
#elif MODE == 1
    // Standard right shift
    asm("shr.u32 %0, %0, %1;" : "+r"(v) : "r"(amt));
#elif MODE == 2
    // Funnel shift left
    asm("shf.l.wrap.b32 %0, %0, %1, %2;" : "+r"(v) : "r"(v), "r"(amt));
#elif MODE == 3
    // PTX vshl 4x8 vector shift (wide-narrow)
    asm("vshl.u32.u32.u32.clamp %0, %0, %1;" : "+r"(v) : "r"(amt));
#elif MODE == 4
    // PTX vshr 2x16 vector shift
    asm("vshr.u32.u32.u32.clamp %0, %0, %1;" : "+r"(v) : "r"(amt));
#endif

    p[idx] = v;
}
