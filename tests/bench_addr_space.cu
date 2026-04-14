// Address space queries + divergent control probes.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
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
    unsigned long long ptr_g = (unsigned long long)A;
    unsigned long long ptr_s = (unsigned long long)&smem[threadIdx.x];
    unsigned int v = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // isspacep.global
            unsigned int r;
            asm volatile("{.reg .pred p; isspacep.global p, %1; selp.u32 %0, 1, 0, p;}" : "=r"(r) : "l"(ptr_g + j));
            v ^= r;
#elif OP == 1  // isspacep.shared
            unsigned int r;
            asm volatile("{.reg .pred p; isspacep.shared p, %1; selp.u32 %0, 1, 0, p;}" : "=r"(r) : "l"(ptr_s + j));
            v ^= r;
#elif OP == 2  // isspacep.local
            unsigned int r;
            asm volatile("{.reg .pred p; isspacep.local p, %1; selp.u32 %0, 1, 0, p;}" : "=r"(r) : "l"(ptr_g + j));
            v ^= r;
#elif OP == 3  // cvta.to.global (generic → global) — may be free
            unsigned long long g;
            asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(g) : "l"(ptr_g + j));
            v ^= (unsigned)g;
#elif OP == 4  // cvta.to.shared (generic → shared)
            unsigned int s;
            asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(*(unsigned long long*)&s) : "l"(ptr_s + j));
            v ^= s;
#elif OP == 5  // setp + branch (divergent)
            unsigned int r;
            if (threadIdx.x & (1u << (j & 4))) r = j;
            else r = ~j;
            v ^= r;
#elif OP == 6  // bar.warp.sync (warp-only sync)
            asm volatile("bar.warp.sync -1;");
            v ^= j;
#elif OP == 7  // setmaxnreg
            asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
            v ^= j;
#elif OP == 8  // ELECT.SYNC
            unsigned int leader_tid;
            unsigned int is_leader;
            asm volatile("{.reg .pred p; elect.sync %0|p, 0xFFFFFFFF; selp.u32 %1, 1, 0, p;}"
                         : "=r"(leader_tid), "=r"(is_leader));
            v ^= is_leader * leader_tid;
#elif OP == 9  // __threadfence_block (membar.cta alternative)
            asm volatile("membar.cta;");
            v ^= j;
#elif OP == 10  // __threadfence (membar.gl)
            asm volatile("membar.gl;");
            v ^= j;
#elif OP == 11  // __threadfence_system (membar.sys)
            asm volatile("membar.sys;");
            v ^= j;
#endif
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
