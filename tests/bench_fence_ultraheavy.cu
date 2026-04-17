// VERY heavy write load to find sc vs acq_rel divergence if any.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#ifndef NWRITES
#define NWRITES 32
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned* my_addr = A + tid * NWRITES;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    unsigned acc = seed;
    #pragma unroll 1
    for (int i = 0; i < 100; i++) {
        #pragma unroll
        for (int j = 0; j < NWRITES; j++) ((volatile unsigned*)my_addr)[j] = i + seed + j;
#if OP == 0
        asm volatile("fence.sc.gpu;");
#elif OP == 1
        asm volatile("fence.acq_rel.gpu;");
#elif OP == 2
        asm volatile("fence.sc.sys;");
#elif OP == 3
        asm volatile("fence.acq_rel.sys;");
#endif
        acc ^= i;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = acc;
    }
}
