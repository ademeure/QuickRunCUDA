// popc.b32 vs popc.b64 throughput - is 64-bit emulated?
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v32 = (unsigned)threadIdx.x + (unsigned)u2;
    unsigned long long v64 = (unsigned long long)v32 * 0x9E3779B97F4A7C15ull;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        unsigned int p;
        asm("popc.b32 %0, %1;" : "=r"(p) : "r"(v32));
        v32 = p ^ v32;
#elif MODE == 1
        unsigned int p;
        asm("popc.b64 %0, %1;" : "=r"(p) : "l"(v64));
        v64 = (unsigned long long)p ^ v64;
#elif MODE == 2
        // 2x popc.b32 to compare to popc.b64
        unsigned int p1, p2;
        asm("popc.b32 %0, %1;" : "=r"(p1) : "r"((unsigned)v64));
        asm("popc.b32 %0, %1;" : "=r"(p2) : "r"((unsigned)(v64 >> 32)));
        v64 = (unsigned long long)(p1 + p2) ^ v64;
#endif
    }

    if (v32 == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v32 + (unsigned)v64;
}
