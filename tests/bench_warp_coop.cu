// Warp cooperative primitives: vote.all/any/ballot, match, __shfl, shfl.up/down/bfly.

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
#if OP == 0  // vote.sync.ballot
            asm volatile("{.reg .pred p; setp.ne.u32 p, %0, 0; vote.sync.ballot.b32 %0, p, 0xFFFFFFFF;}" : "+r"(v));
#elif OP == 1  // vote.sync.all
            unsigned int p;
            asm volatile("{.reg .pred q,r; setp.ne.u32 q, %1, 0; vote.sync.all.pred r, q, 0xFFFFFFFF; selp.u32 %0, 1, 0, r;}" : "=r"(p) : "r"(v));
            v ^= p;
#elif OP == 2  // vote.sync.any
            unsigned int p;
            asm volatile("{.reg .pred q,r; setp.ne.u32 q, %1, 0; vote.sync.any.pred r, q, 0xFFFFFFFF; selp.u32 %0, 1, 0, r;}" : "=r"(p) : "r"(v));
            v ^= p;
#elif OP == 3  // vote.sync.uni
            unsigned int p;
            asm volatile("{.reg .pred q,r; setp.ne.u32 q, %1, 0; vote.sync.uni.pred r, q, 0xFFFFFFFF; selp.u32 %0, 1, 0, r;}" : "=r"(p) : "r"(v));
            v ^= p;
#elif OP == 4  // shfl.sync.bfly with small range
            asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1F, -1;" : "+r"(v));
#elif OP == 5  // shfl.sync.idx (broadcast lane 0)
            asm volatile("shfl.sync.idx.b32 %0, %0, 0, 0x1F, -1;" : "+r"(v));
#elif OP == 6  // shfl.sync.up by 1
            asm volatile("shfl.sync.up.b32 %0, %0, 1, 0x00, -1;" : "+r"(v));
#elif OP == 7  // shfl.sync.down by 1
            asm volatile("shfl.sync.down.b32 %0, %0, 1, 0x1F, -1;" : "+r"(v));
#elif OP == 8  // match.all (warp uniform — fast path?)
            unsigned int r;
            unsigned int p;
            asm volatile("{.reg .pred q; match.all.sync.b32 %0|q, %1, 0xFFFFFFFF; selp.u32 %2, 1, 0, q;}"
                : "=r"(r), "=r"(p) : "r"(v));
            v = r ^ p;
#elif OP == 9  // redux.sync.min (warp-wide reduction)
            asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v));
#elif OP == 10  // redux.sync.add
            asm volatile("redux.sync.add.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v));
#elif OP == 11  // redux.sync.or
            asm volatile("redux.sync.or.b32 %0, %0, 0xFFFFFFFF;" : "+r"(v));
#endif
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
