// Deeper 2-hotspot anomaly investigation.
// Vary WITHIN-WARP distribution of hotspots.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef LAYOUT
#define LAYOUT 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = threadIdx.x & 0x1F;
    unsigned int addr_idx;

#if LAYOUT == 0
    addr_idx = 0;                               // all same
#elif LAYOUT == 1
    addr_idx = lane & 1;                         // 2 hotspots, interleaved within warp
#elif LAYOUT == 2
    addr_idx = (lane < 16) ? 0 : 1;              // 2 hotspots, half-warp split
#elif LAYOUT == 3
    addr_idx = lane & 3;                         // 4 hotspots interleaved
#elif LAYOUT == 4
    addr_idx = lane / 8;                          // 4 hotspots, 8-lane groups
#elif LAYOUT == 5
    addr_idx = tid % 2;                           // 2 hotspots, warps alternate
#elif LAYOUT == 6
    // Each warp hits 1 hotspot (different warp, different hotspot)
    addr_idx = (threadIdx.x >> 5) & 0x1;
#elif LAYOUT == 7
    addr_idx = lane;                             // 32 hotspots = unique per lane
#elif LAYOUT == 8
    addr_idx = tid;                              // unique per thread
#endif

    unsigned long long addr = (unsigned long long)(A + addr_idx);
    unsigned int v = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned int r;
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(addr), "r"(1u));
            v ^= r;
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[tid] = v;
}
