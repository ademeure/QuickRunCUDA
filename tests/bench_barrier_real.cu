// REAL __syncthreads cost when thread arrival is staggered.
// Single thread arrives early → waits for others. Measure different patterns.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = threadIdx.x;
    unsigned long long t_start = 0, t_end = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        // record time at start of group
        if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t_start));
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // plain __syncthreads — all threads arrive synchronously
            __syncthreads();
#elif OP == 1  // staggered: half warps do FFMA chain first
            if ((threadIdx.x >> 5) & 1) {
                float f = 0.1f;
                for (int k = 0; k < 20; k++) f = f * 1.001f + 0.001f;
                v += __float_as_int(f);
            }
            __syncthreads();
#elif OP == 2  // severe stagger: only lane 0 of warp 0 does heavy work
            if (threadIdx.x == 0) {
                float f = 0.1f;
                for (int k = 0; k < 200; k++) f = f * 1.001f + 0.001f;
                v += __float_as_int(f);
            }
            __syncthreads();
#elif OP == 3  // warp.sync only (no block barrier)
            __syncwarp();
#elif OP == 4  // bar.arrive (non-blocking arrive, wait elsewhere)
            asm volatile("bar.arrive 0, %0;" :: "r"(BLOCK_SIZE));
            asm volatile("bar.sync 0;");
#endif
        }
        if (threadIdx.x == 0) {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t_end));
            v = (unsigned)(t_end - t_start);
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
