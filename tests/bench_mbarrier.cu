// mbarrier (memory-backed barrier) and DEPBAR probes.

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

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // mbarrier needs 64-bit aligned storage
    unsigned long long* mb = (unsigned long long*)&smem[0];
    if (threadIdx.x == 0) {
        unsigned int addr = (unsigned)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(addr));
    }
    __syncthreads();

    unsigned int addr = (unsigned)__cvta_generic_to_shared(mb);
    unsigned long long tok = 0;
    unsigned int v = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // mbarrier.arrive - just arrive (no wait)
            asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(tok) : "r"(addr));
            v ^= (unsigned)tok;
#elif OP == 1  // mbarrier.arrive_drop (also drops participant count)
            asm volatile("mbarrier.arrive_drop.shared.b64 %0, [%1];" : "=l"(tok) : "r"(addr));
            v ^= (unsigned)tok;
#elif OP == 2  // mbarrier.test_wait (non-blocking test)
            unsigned int pass;
            asm volatile("{.reg .pred p; mbarrier.test_wait.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
                         : "=r"(pass) : "r"(addr), "l"(tok));
            v ^= pass;
#elif OP == 3  // mbarrier.inval (invalidate)
            if (threadIdx.x == 0) asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(addr));
            v ^= j;
#elif OP == 4  // cp.async + commit_group + wait_group pattern (staged async)
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_all;");
            v ^= j;
#elif OP == 5  // DEPBAR — explicit dep barrier
            asm volatile("bar.cta.arrive;");
            asm volatile("bar.cta.wait;");
#endif
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
