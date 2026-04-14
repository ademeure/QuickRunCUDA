// Compare warp-reduce via redux.sync vs manual shfl.sync tree.

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

__device__ __forceinline__ unsigned int warp_min_shfl(unsigned int v) {
    for (int k = 16; k > 0; k >>= 1) {
        unsigned int o;
        asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1F, -1;" : "=r"(o) : "r"(v), "r"(k));
        v = min(v, o);
    }
    return v;
}
__device__ __forceinline__ unsigned int warp_add_shfl(unsigned int v) {
    for (int k = 16; k > 0; k >>= 1) {
        unsigned int o;
        asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1F, -1;" : "=r"(o) : "r"(v), "r"(k));
        v = v + o;
    }
    return v;
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = threadIdx.x + 1;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // redux.sync.min.u32 (hw)
            asm volatile("redux.sync.min.u32 %0, %0, -1;" : "+r"(v));
#elif OP == 1  // shfl tree min
            v = warp_min_shfl(v);
#elif OP == 2  // redux.sync.add.u32 (hw)
            asm volatile("redux.sync.add.u32 %0, %0, -1;" : "+r"(v));
#elif OP == 3  // shfl tree add
            v = warp_add_shfl(v);
#endif
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
