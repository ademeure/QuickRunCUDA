// TMEM peak: 1 warpgroup (4 warps, 128 threads) per CTA. Each warp accesses
// its own TMEM partition. Read AND write tested.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef ITERS
#define ITERS 100
#endif
#ifndef OP
#define OP 0
#endif
#ifndef CHAINS
#define CHAINS 8
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    __shared__ __align__(4) unsigned tmem_slot;
    if (threadIdx.x == 0) tmem_slot = 0xFFFFFFFF;
    __syncthreads();

    if (threadIdx.x < 32) {
        // Only warp 0 allocates: 256 cols (covers all 4 partitions)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;"
            :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    }
    __syncthreads();
    unsigned tmem_addr = tmem_slot;
    unsigned warp_id = threadIdx.x >> 5;  // 0..3

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    unsigned acc = 0;
#if OP == 0
    // All warps use same base — HW automatically assigns per-warp partition
    unsigned my_base = tmem_addr;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned r[16];
        // Single load per iter, no offset
        asm volatile("tcgen05.ld.sync.aligned.16x64b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
              "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
              "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
              "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
            : "r"(my_base));
        for (int j = 0; j < 16; j++) acc ^= r[j];
    }
#elif OP == 1
    // tcgen05.st (write) - same pattern
    unsigned my_base = tmem_addr;
    unsigned vals[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) vals[i] = (seed ^ threadIdx.x) + i;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("tcgen05.st.sync.aligned.16x64b.x16.b32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
            :: "r"(my_base),
               "r"(vals[0]),"r"(vals[1]),"r"(vals[2]),"r"(vals[3]),
               "r"(vals[4]),"r"(vals[5]),"r"(vals[6]),"r"(vals[7]),
               "r"(vals[8]),"r"(vals[9]),"r"(vals[10]),"r"(vals[11]),
               "r"(vals[12]),"r"(vals[13]),"r"(vals[14]),"r"(vals[15]) : "memory");
    }
    // Force a wait so the stores complete
    asm volatile("tcgen05.wait::st.sync.aligned;");
    acc = vals[0] ^ vals[1];
#endif

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem_addr));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned*)C)[2] = acc;
    }
}
