// ULDC (uniform datapath) vs scalar LDC throughput
extern "C" __constant__ unsigned int CMEM[1024];

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)threadIdx.x + (unsigned)u2;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Uniform addr (all threads same idx) → ULDC
        unsigned int idx = (i + (unsigned)u2) & 0x3FF;
        unsigned int x;
        asm("ld.const.u32 %0, [%1];" : "=r"(x) : "l"(CMEM + idx));
        v ^= x;
#elif MODE == 1
        // Per-thread addr → LDC (scalar constant load)
        unsigned int idx = (threadIdx.x + i + (unsigned)u2) & 0x3FF;
        unsigned int x;
        asm("ld.const.u32 %0, [%1];" : "=r"(x) : "l"(CMEM + idx));
        v ^= x;
#elif MODE == 2
        // Mixed: uniform within warp, varies per warp
        unsigned int idx = ((blockIdx.x + i) & 0x3FF);
        unsigned int x;
        asm("ld.const.u32 %0, [%1];" : "=r"(x) : "l"(CMEM + idx));
        v ^= x;
#endif
    }

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
