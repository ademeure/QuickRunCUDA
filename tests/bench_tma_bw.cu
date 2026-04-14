// cp.async.bulk (TMA) throughput — how fast can one SM pull from DRAM via TMA?

#ifndef UNROLL
#define UNROLL 8
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
#ifndef BYTES
#define BYTES 128
#endif

extern __shared__ __align__(16) unsigned char smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned long long* bar = (unsigned long long*)smem;
    unsigned char* data = smem + 16;

    if (threadIdx.x == 0) {
        unsigned int bar_addr = (unsigned)__cvta_generic_to_shared(bar);
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar_addr), "r"(1u));
    }
    __syncthreads();

    unsigned int bar_addr = (unsigned)__cvta_generic_to_shared(bar);
    unsigned int data_addr = (unsigned)__cvta_generic_to_shared(data);
    unsigned long long src = (unsigned long long)A + blockIdx.x * 4096;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // cp.async.bulk (128B per call, thread 0)
            if (threadIdx.x == 0) {
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                    :: "r"(data_addr), "l"(src + i * 128), "n"(BYTES), "r"(bar_addr));
            }
#elif OP == 1  // cp.async.cg (per-thread 16-byte)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                :: "r"(data_addr + threadIdx.x*16), "l"(src + threadIdx.x*16 + i*16));
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_all;");
#endif
        }
    }
    if (threadIdx.x == 0 && bar_addr == (unsigned)seed) ((unsigned int*)C)[0] = data_addr;
}
