// Clean CREDUX test — single chain, u32 register, no array indexing IMADs.

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef OP
#define OP 0   // 0=min, 1=max, 2=add, 3=or
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Independent register chains (not array → no index IMADs)
    unsigned int v0 = 0xDEADBEEFu ^ (threadIdx.x * 131 + 0 * 17);
    unsigned int v1 = 0xDEADBEEFu ^ (threadIdx.x * 131 + 1 * 17);
    unsigned int v2 = 0xDEADBEEFu ^ (threadIdx.x * 131 + 2 * 17);
    unsigned int v3 = 0xDEADBEEFu ^ (threadIdx.x * 131 + 3 * 17);
    unsigned int v4 = 0xDEADBEEFu ^ (threadIdx.x * 131 + 4 * 17);
    unsigned int v5 = 0xDEADBEEFu ^ (threadIdx.x * 131 + 5 * 17);
    unsigned int v6 = 0xDEADBEEFu ^ (threadIdx.x * 131 + 6 * 17);
    unsigned int v7 = 0xDEADBEEFu ^ (threadIdx.x * 131 + 7 * 17);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0
            asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v0));
            asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v1));
            asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v2));
            asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v3));
            asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v4));
            asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v5));
            asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v6));
            asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v7));
#elif OP == 1
            asm volatile("redux.sync.max.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v0));
            asm volatile("redux.sync.max.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v1));
            asm volatile("redux.sync.max.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v2));
            asm volatile("redux.sync.max.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v3));
            asm volatile("redux.sync.max.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v4));
            asm volatile("redux.sync.max.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v5));
            asm volatile("redux.sync.max.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v6));
            asm volatile("redux.sync.max.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v7));
#elif OP == 2
            asm volatile("redux.sync.add.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v0));
            asm volatile("redux.sync.add.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v1));
            asm volatile("redux.sync.add.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v2));
            asm volatile("redux.sync.add.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v3));
            asm volatile("redux.sync.add.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v4));
            asm volatile("redux.sync.add.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v5));
            asm volatile("redux.sync.add.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v6));
            asm volatile("redux.sync.add.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v7));
#elif OP == 3
            asm volatile("redux.sync.or.b32 %0, %0, 0xFFFFFFFF;" : "+r"(v0));
            asm volatile("redux.sync.or.b32 %0, %0, 0xFFFFFFFF;" : "+r"(v1));
            asm volatile("redux.sync.or.b32 %0, %0, 0xFFFFFFFF;" : "+r"(v2));
            asm volatile("redux.sync.or.b32 %0, %0, 0xFFFFFFFF;" : "+r"(v3));
            asm volatile("redux.sync.or.b32 %0, %0, 0xFFFFFFFF;" : "+r"(v4));
            asm volatile("redux.sync.or.b32 %0, %0, 0xFFFFFFFF;" : "+r"(v5));
            asm volatile("redux.sync.or.b32 %0, %0, 0xFFFFFFFF;" : "+r"(v6));
            asm volatile("redux.sync.or.b32 %0, %0, 0xFFFFFFFF;" : "+r"(v7));
#endif
        }
    }
    unsigned int acc = v0 ^ v1 ^ v2 ^ v3 ^ v4 ^ v5 ^ v6 ^ v7;
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
