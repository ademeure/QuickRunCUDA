// Special register reads + clock overhead + special sync ops.

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

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = threadIdx.x;
    unsigned long long t = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // %clock (SR read)
            unsigned int c;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
            v ^= c;
#elif OP == 1  // %clock64 (64-bit)
            unsigned long long c;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
            v ^= (unsigned)c;
            t ^= c;
#elif OP == 2  // %globaltimer
            unsigned long long c;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(c));
            t ^= c;
#elif OP == 3  // %gridid
            unsigned long long g;
            asm volatile("mov.u64 %0, %%gridid;" : "=l"(g));
            t ^= g;
#elif OP == 4  // %smid (which SM)
            unsigned int s;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
            v ^= s;
#elif OP == 5  // %warpid (local)
            unsigned int w;
            asm volatile("mov.u32 %0, %%warpid;" : "=r"(w));
            v ^= w;
#elif OP == 6  // %nwarpid (# warps per SM)
            unsigned int w;
            asm volatile("mov.u32 %0, %%nwarpid;" : "=r"(w));
            v ^= w;
#elif OP == 7  // %envreg0..7 (driver-provided)
            unsigned int e;
            asm volatile("mov.u32 %0, %%envreg0;" : "=r"(e));
            v ^= e;
#elif OP == 8  // %lanemask_eq
            unsigned int m;
            asm volatile("mov.u32 %0, %%lanemask_eq;" : "=r"(m));
            v ^= m;
#elif OP == 9  // %pm0 (perf monitor)
            unsigned int p;
            asm volatile("mov.u32 %0, %%pm0;" : "=r"(p));
            v ^= p;
#elif OP == 10 // %clusterid.x
            unsigned int c;
            asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(c));
            v ^= c;
#elif OP == 11 // getctarank
            unsigned int r;
            asm volatile("getctarank.u32 %0, 0;" : "=r"(r));
            v ^= r;
#endif
        }
    }
    if ((int)(v ^ (unsigned)t) == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
