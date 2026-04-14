// Reconvergence barriers (BSSY/BSYNC) and divergent control cost.

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

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = threadIdx.x;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // divergent if (half lanes)
            if (threadIdx.x & 16) {
                v = v * 3 + 1;
            } else {
                v = v * 5 + 2;
            }
#elif OP == 1  // non-divergent if (all lanes take same branch)
            if (blockIdx.x & 1) {
                v = v * 3 + 1;
            } else {
                v = v * 5 + 2;
            }
#elif OP == 2  // small divergent loop (warp-invariant count)
            for (int kk = 0; kk < (blockIdx.x & 3); kk++) v = v * 3 + 1;
#elif OP == 3  // divergent switch
            switch (threadIdx.x & 3) {
                case 0: v = v * 3 + 1; break;
                case 1: v = v * 5 + 2; break;
                case 2: v = v ^ 0xAB; break;
                case 3: v = v + j; break;
            }
#elif OP == 4  // setp + selp (data-dep, no branch)
            unsigned int a = v * 3 + 1, b = v * 5 + 2;
            asm volatile("{.reg .pred p; setp.lt.u32 p, %0, 0x80000000; selp.u32 %0, %1, %2, p;}" : "+r"(v) : "r"(a), "r"(b));
#elif OP == 5  // Uniform branch (entire warp takes same path each time)
            if (blockIdx.x & j & 1) v += 1;
            else v += 2;
#endif
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
