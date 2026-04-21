// PRMT byte permute modes
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)threadIdx.x + (unsigned)u2;
    unsigned int b = 0xDEADBEEFu + (unsigned)u2;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Default prmt (mode 0)
        asm("prmt.b32 %0, %0, %1, 0x7654;" : "+r"(v) : "r"(b));
#elif MODE == 1
        // prmt.f4e (forward 4-extract)
        asm("prmt.b32.f4e %0, %0, %1, 0x3210;" : "+r"(v) : "r"(b));
#elif MODE == 2
        // prmt.b4e (backward 4-extract)
        asm("prmt.b32.b4e %0, %0, %1, 0x3210;" : "+r"(v) : "r"(b));
#elif MODE == 3
        // prmt.rc8 (replicate component 8)
        asm("prmt.b32.rc8 %0, %0, %1, 0x3210;" : "+r"(v) : "r"(b));
#elif MODE == 4
        // prmt.ecl (edge clamp left)
        asm("prmt.b32.ecl %0, %0, %1, 0x3210;" : "+r"(v) : "r"(b));
#elif MODE == 5
        // prmt.ecr (edge clamp right)
        asm("prmt.b32.ecr %0, %0, %1, 0x3210;" : "+r"(v) : "r"(b));
#elif MODE == 6
        // prmt.rc16 (replicate component 16)
        asm("prmt.b32.rc16 %0, %0, %1, 0x3210;" : "+r"(v) : "r"(b));
#endif
    }

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
