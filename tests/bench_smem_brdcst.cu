// Shared memory broadcast pattern — all warp lanes read same SMEM addr.
// Is there a broadcast fast path?

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
#define OP 0
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int tid = threadIdx.x;
    if (tid < 1024) smem[tid] = tid * 0x11 + 1;
    __syncthreads();

    unsigned int v = 0;
    unsigned int addr_same = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned int addr_diff = (unsigned)__cvta_generic_to_shared(&smem[tid & 0x1F]);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0   // BROADCAST: all 32 lanes same addr (fast path)
            unsigned int r;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(addr_same));
            v ^= r;
#elif OP == 1 // Per-lane unique addr (bank-clean)
            unsigned int r;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(addr_diff));
            v ^= r;
#elif OP == 2 // Two-warp broadcast: half-warp same, half-warp different
            unsigned int addr = (tid & 0x10) ? addr_same : addr_diff;
            unsigned int r;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(addr));
            v ^= r;
#elif OP == 3 // 4-way bank conflict
            unsigned int addr = (unsigned)__cvta_generic_to_shared(&smem[(tid & 7) * 4]);
            unsigned int r;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(addr));
            v ^= r;
#elif OP == 4 // 32-way bank conflict (all same bank)
            unsigned int addr = (unsigned)__cvta_generic_to_shared(&smem[(tid & 1) * 32]);
            unsigned int r;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(addr));
            v ^= r;
#endif
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + tid] = v;
}
