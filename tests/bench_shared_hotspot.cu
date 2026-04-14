// Shared-memory hotspot atomics — does the warp-coalesce happen here too?

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef LAYOUT
#define LAYOUT 0
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 0x1F;
    unsigned int addr_idx;

#if LAYOUT == 0   // all same (1 hotspot)
    addr_idx = 0;
#elif LAYOUT == 1 // within-warp 2 hotspots (lane%2)
    addr_idx = lane & 1;
#elif LAYOUT == 2 // each warp same, different warps different (warp%2)
    addr_idx = (tid >> 5) & 0x1;
#elif LAYOUT == 3 // unique per lane
    addr_idx = lane;
#elif LAYOUT == 4 // unique per thread (no contention in block)
    addr_idx = tid & 0x3F;
#endif

    if (tid < 64) smem[tid] = 0;
    __syncthreads();

    unsigned int base = (unsigned)__cvta_generic_to_shared(&smem[addr_idx]);
    unsigned int v = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned int r;
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(base), "r"(1u));
            v ^= r;
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + tid] = v;
}
