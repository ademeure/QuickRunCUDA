// cp.async.bulk TMA probe — bulk copy from global to shared via TMA.
// Needs a TMA descriptor (we build a simple one on-the-fly).

#ifndef UNROLL
#define UNROLL 16
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

extern __shared__ __align__(16) unsigned char smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Smem region for mbarrier + data
    unsigned long long* bar = (unsigned long long*)smem;
    unsigned char* data = smem + 16;

    if (threadIdx.x == 0) {
        unsigned int bar_addr = (unsigned)__cvta_generic_to_shared(bar);
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar_addr), "r"(1u));
    }
    __syncthreads();

    unsigned int bar_addr = (unsigned)__cvta_generic_to_shared(bar);
    unsigned int data_addr = (unsigned)__cvta_generic_to_shared(data);
    unsigned long long src = (unsigned long long)A;

    unsigned long long tok = 0;
    unsigned int v = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // cp.async.bulk (128 bytes from global to shared via mbarrier)
            if (threadIdx.x == 0) {
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];"
                    :: "r"(data_addr), "l"(src), "r"(bar_addr));
            }
#elif OP == 1  // cp.async (regular, 16 bytes)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                :: "r"(data_addr + (threadIdx.x * 16)), "l"(src + threadIdx.x * 16));
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_all;");
#elif OP == 2  // regular global load (for comparison)
            unsigned int r;
            asm volatile("ld.global.u32 %0, [%1];" : "=r"(r) : "l"(src + threadIdx.x * 4));
            v ^= r;
#endif
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v + (unsigned)tok;
}
