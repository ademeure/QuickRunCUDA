// Comprehensive matrix: variable SMs × variable writes per iter × membar.sys cost.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ACTIVE_BLOCKS
#define ACTIVE_BLOCKS 1
#endif
#ifndef NWRITES
#define NWRITES 0
#endif
#ifndef ITERS
#define ITERS 100
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    if (blockIdx.x >= ACTIVE_BLOCKS) return;
    if (threadIdx.x != 0) return;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned* my_addr = A + tid * 32;

    unsigned long long t0, t1;
    unsigned total = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
#if NWRITES > 0
        #pragma unroll
        for (int j = 0; j < NWRITES; j++) {
            ((volatile unsigned*)my_addr)[j] = i + seed + j;
        }
#endif
#ifdef USE_GL
        asm volatile("membar.gl;");
#elif defined(USE_CTA)
        asm volatile("membar.cta;");
#else
        asm volatile("membar.sys;");
#endif
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total += (unsigned)(t1 - t0);
    }
    C[blockIdx.x] = total / ITERS;
}
