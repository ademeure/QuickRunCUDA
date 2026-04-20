// 32KB SMEM -> 1 value reduction in fewest cycles
// 256-thread block, 8192 ints in SMEM, reduce to 1 int
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ unsigned int smem[8192];
    __shared__ unsigned int warp_sums[8];

    // Load SMEM with deterministic data
    for (int i = threadIdx.x; i < 8192; i += blockDim.x) {
        smem[i] = i + (unsigned)u2;
    }
    __syncthreads();

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

#if MODE == 0
    // Per-thread sum 32 elements (8192 / 256), warp reduce, block reduce
    unsigned int sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) sum += smem[i * 256 + threadIdx.x];
    // Warp reduce via SHFL
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum,  8);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum,  4);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum,  2);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum,  1);
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 8) {
        unsigned int s = warp_sums[threadIdx.x];
        s += __shfl_xor_sync(0xFF, s, 4);
        s += __shfl_xor_sync(0xFF, s, 2);
        s += __shfl_xor_sync(0xFF, s, 1);
        if (threadIdx.x == 0) warp_sums[0] = s;
    }
#elif MODE == 1
    // Same but with redux.sync.add (int)
    unsigned int sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) sum += smem[i * 256 + threadIdx.x];
    asm("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(sum) : "r"(sum));
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 8) {
        unsigned int s = warp_sums[threadIdx.x];
        // Use SHFL chain for 8-lane sub-warp reduce
        s += __shfl_xor_sync(0xFF, s, 4);
        s += __shfl_xor_sync(0xFF, s, 2);
        s += __shfl_xor_sync(0xFF, s, 1);
        if (threadIdx.x == 0) warp_sums[0] = s;
    }
#elif MODE == 2
    // vec4 loads + redux.sync per warp
    unsigned int sum = 0;
    uint4* smem4 = (uint4*)smem;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint4 v = smem4[i * 256 + threadIdx.x];
        sum += v.x + v.y + v.z + v.w;
    }
    asm("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(sum) : "r"(sum));
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 8) {
        unsigned int s = warp_sums[threadIdx.x];
        s += __shfl_xor_sync(0xFF, s, 4);
        s += __shfl_xor_sync(0xFF, s, 2);
        s += __shfl_xor_sync(0xFF, s, 1);
        if (threadIdx.x == 0) warp_sums[0] = s;
    }
#endif

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (threadIdx.x == 0) C[blockIdx.x] = (float)warp_sums[0];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu sum=%u\n", MODE, t1 - t0, warp_sums[0]);
    }
}
