// mov.b64 wide register copy
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned long long lv = ((unsigned long long)threadIdx.x << 32) | (unsigned)u2;
    unsigned long long lb = lv ^ 0xDEADBEEFCAFEBABEull;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Pure 64-bit add chain
        lv = lv * 31u + lb;
#elif MODE == 1
        // mov.b64 then add
        unsigned long long t;
        asm("mov.b64 %0, %1;" : "=l"(t) : "l"(lb));
        lv = lv * 31u + t;
#elif MODE == 2
        // Two mov.b32 then OR
        unsigned int hi = (unsigned)(lb >> 32);
        unsigned int lo = (unsigned)lb;
        asm("mov.b32 %0, %1;" : "=r"(hi) : "r"(hi));
        asm("mov.b32 %0, %1;" : "=r"(lo) : "r"(lo));
        unsigned long long t = ((unsigned long long)hi << 32) | lo;
        lv = lv * 31u + t;
#elif MODE == 3
        // mov.b64 with pair-pack via mov.b64 {a,b}, %1
        unsigned int hi = (unsigned)(lb >> 32);
        unsigned int lo = (unsigned)lb;
        asm("mov.b64 %0, {%1, %2};" : "=l"(lb) : "r"(lo), "r"(hi));
        lv = lv * 31u + lb;
#endif
    }

    if ((int)lv == seed) ((unsigned long long*)C)[blockIdx.x * blockDim.x + threadIdx.x] = lv;
}
