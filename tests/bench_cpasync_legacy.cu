// Legacy cp.async (pre-TMA) throughput & latency, smaller granularity.
// Each thread issues 4/8/16 bytes per instruction.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BYTES
#define BYTES 16
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned long long off = ((unsigned long long)(i + j) * BYTES * BLOCK_SIZE
                                    + threadIdx.x * BYTES) & 0x3FFFFFFFull;
            unsigned long long src = (unsigned long long)A + off;
            unsigned slot = smem_addr + threadIdx.x * BYTES + ((i + j) & 63) * BYTES * BLOCK_SIZE;
            slot = slot & 0xFFFF;
#if OP == 0  // cp.async.ca.shared::cta.global (cacheable via L1)
            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], %2;"
                :: "r"(slot), "l"(src), "n"((unsigned)BYTES) : "memory");
#elif OP == 1  // cp.async.cg (no L1 cache; only 16B)
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], %2;"
                :: "r"(slot), "l"(src), "n"((unsigned)BYTES) : "memory");
#endif
            asm volatile("cp.async.commit_group;");
        }
    }
    asm volatile("cp.async.wait_all;");
    __syncthreads();
    unsigned x;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem_addr + threadIdx.x * 4));
    if ((int)x == seed) ((unsigned int*)C)[blockIdx.x * BLOCK_SIZE + threadIdx.x] = x;
}
