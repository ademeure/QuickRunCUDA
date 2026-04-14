// Reducing load (LDGMC — new Blackwell) + LDGSTS queue depth.

#ifndef UNROLL
#define UNROLL 8
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
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int v = 0;
    unsigned int sbase = (unsigned)__cvta_generic_to_shared(&smem[threadIdx.x]);
    unsigned long long gbase = (unsigned long long)(A + tid * 8);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // ld.global.nc with a prefetch hint
            unsigned int r;
            asm volatile("ld.global.L2::256B.u32 %0, [%1];" : "=r"(r) : "l"(gbase));
            v ^= r;
#elif OP == 1  // standard ld.global.nc
            unsigned int r;
            asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(r) : "l"(gbase));
            v ^= r;
#elif OP == 2  // Burst of cp.async to fill queue
            #pragma unroll
            for (int k=0;k<4;k++) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(sbase + k*16*(BLOCK_SIZE)), "l"(gbase + k*16));
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_all;");
#elif OP == 3  // Burst of cp.async with L2::256B hint
            #pragma unroll
            for (int k=0;k<4;k++) {
                asm volatile("cp.async.ca.shared.global.L2::256B [%0], [%1], 16;"
                    :: "r"(sbase + k*16*(BLOCK_SIZE)), "l"(gbase + k*16));
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_all;");
#elif OP == 4  // Multiple commit_group without wait (queue depth)
            #pragma unroll
            for (int k=0;k<4;k++) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(sbase + k*16*(BLOCK_SIZE)), "l"(gbase + k*16));
                asm volatile("cp.async.commit_group;");
            }
            asm volatile("cp.async.wait_group 3;");
#endif
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[tid] = v;
}
