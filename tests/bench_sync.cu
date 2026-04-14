// Barriers / syncs: cost of __syncwarp / __syncthreads variants at different scales.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = threadIdx.x;
    smem[threadIdx.x] = v;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // __syncwarp
            asm volatile("bar.warp.sync -1;");
#elif OP == 1  // __syncthreads (bar.sync 0)
            asm volatile("bar.sync 0;");
#elif OP == 2  // bar.sync with named resource 1
            asm volatile("bar.sync 1;");
#elif OP == 3  // __threadfence_block + bar.sync
            asm volatile("membar.cta;");
            asm volatile("bar.sync 0;");
#elif OP == 4  // warpsync (llvm intrinsic)
            asm volatile("barrier.sync.aligned 0;");
#elif OP == 5  // arrive + wait on named bar
            asm volatile("bar.sync 2, 128;");
#elif OP == 6  // __syncwarp with specific mask (8 threads active)
            asm volatile("bar.warp.sync 0x000000FF;");
#elif OP == 7  // no-op baseline
            v += j;
#endif
        }
    }
    if (v == (unsigned)seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v + smem[threadIdx.x];
}
