// cp.async completion latency — time from issue to wait-group-N complete.

#ifndef UNROLL
#define UNROLL 4
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
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
    unsigned int tid = threadIdx.x;
    unsigned int base_s = (unsigned)__cvta_generic_to_shared(&smem[tid * 4]);
    unsigned long long base_g = (unsigned long long)(A + (blockIdx.x * blockDim.x + tid) * 16);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // cp.async + commit + wait_all
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                :: "r"(base_s), "l"(base_g));
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_all;");
#elif OP == 1  // cp.async x4 + commit + wait_all (pipelined amortize)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(base_s+0), "l"(base_g+0));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(base_s+16), "l"(base_g+16));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(base_s+32), "l"(base_g+32));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(base_s+48), "l"(base_g+48));
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_all;");
#elif OP == 2  // cp.async + commit + wait_group N=0 (non-blocking)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                :: "r"(base_s), "l"(base_g));
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");  // wait until 1 group left (non-blocking for last)
#elif OP == 3  // 2-stage pipeline: commit, overlap with compute, wait
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                :: "r"(base_s), "l"(base_g));
            asm volatile("cp.async.commit_group;");
            // compute in gap (does nothing useful but models overlap)
            float f = (float)j;
            f = f * 1.001f + 0.001f;
            asm volatile("cp.async.wait_all;");
            if (f == 0.0f) smem[tid] = 0;  // keep f live
#elif OP == 4  // synchronous LDG.128 (for comparison)
            unsigned int x0,x1,x2,x3;
            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "l"(base_g));
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(base_s), "r"(x0),"r"(x1),"r"(x2),"r"(x3));
#endif
        }
    }
    if ((int)smem[tid] == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = smem[tid];
}
