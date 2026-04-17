// Detailed fence.sc vs fence.acq_rel comparison across scopes, traffic levels.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ACTIVE_BLOCKS
#define ACTIVE_BLOCKS 1
#endif
#ifndef NWRITES
#define NWRITES 0
#endif
#ifndef OP
#define OP 0
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
    for (int i = 0; i < 100; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
#if NWRITES > 0
        #pragma unroll
        for (int j = 0; j < NWRITES; j++) ((volatile unsigned*)my_addr)[j] = i + seed + j;
#endif
#if OP == 0
        asm volatile("fence.sc.cta;");
#elif OP == 1
        asm volatile("fence.acq_rel.cta;");
#elif OP == 2
        asm volatile("fence.sc.gpu;");
#elif OP == 3
        asm volatile("fence.acq_rel.gpu;");
#elif OP == 4
        asm volatile("fence.sc.sys;");
#elif OP == 5
        asm volatile("fence.acq_rel.sys;");
#endif
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total += (unsigned)(t1 - t0);
    }
    C[blockIdx.x] = total / 100;
}
