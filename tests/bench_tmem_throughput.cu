// TMEM throughput: many parallel reads to saturate BW.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ITERS
#define ITERS 100
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    __shared__ __align__(4) unsigned tmem_slot;
    if (threadIdx.x == 0) tmem_slot = 0xFFFFFFFF;
    __syncthreads();

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    __syncthreads();
    unsigned tmem_addr = tmem_slot;

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Many independent x16 loads (each 16x64b = 2048 B per warp)
    unsigned acc = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned r[16];
        // 8 independent loads (8 chains × x16 = 16384 B per inner iter)
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            asm volatile("tcgen05.ld.sync.aligned.16x64b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
                  "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                  "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
                  "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
                : "r"(tmem_addr + k * 16));
            for (int j = 0; j < 16; j++) acc ^= r[j];
        }
    }

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem_addr));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");

    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned*)C)[2] = acc;
    }
}
