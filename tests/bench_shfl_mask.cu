// Does SHFL rate depend on warp mask width?

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef MASK
#define MASK 0xFFFFFFFF
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v0=threadIdx.x,v1=threadIdx.x*3,v2=threadIdx.x*5,v3=threadIdx.x*7;
    unsigned int v4=threadIdx.x*11,v5=threadIdx.x*13,v6=threadIdx.x*17,v7=threadIdx.x*19;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1F, %1;" : "+r"(v0) : "n"(MASK));
            asm volatile("shfl.sync.bfly.b32 %0, %0, 2, 0x1F, %1;" : "+r"(v1) : "n"(MASK));
            asm volatile("shfl.sync.bfly.b32 %0, %0, 4, 0x1F, %1;" : "+r"(v2) : "n"(MASK));
            asm volatile("shfl.sync.bfly.b32 %0, %0, 8, 0x1F, %1;" : "+r"(v3) : "n"(MASK));
            asm volatile("shfl.sync.bfly.b32 %0, %0, 16, 0x1F, %1;" : "+r"(v4) : "n"(MASK));
            asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1F, %1;" : "+r"(v5) : "n"(MASK));
            asm volatile("shfl.sync.bfly.b32 %0, %0, 2, 0x1F, %1;" : "+r"(v6) : "n"(MASK));
            asm volatile("shfl.sync.bfly.b32 %0, %0, 4, 0x1F, %1;" : "+r"(v7) : "n"(MASK));
        }
    }
    unsigned int acc = v0^v1^v2^v3^v4^v5^v6^v7;
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
