// Test if tcgen05.st drain time is large.
// Compare: N writes + 1 wait vs N writes (no wait) vs 1 write + 1 wait.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef N
#define N 100
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    __shared__ __align__(4) unsigned tmem_slot;
    if (threadIdx.x == 0) tmem_slot = 0xFFFFFFFF;
    __syncthreads();

    if (threadIdx.x < 32) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;"
            :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    }
    __syncthreads();
    unsigned tmem_addr = tmem_slot;

    unsigned vals[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) vals[i] = (threadIdx.x << 16) | (seed + i);

    unsigned long long t0, t1, t2;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

#if OP == 0
    // N writes, then ONE wait at end
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        asm volatile("tcgen05.st.sync.aligned.16x64b.x16.b32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
            :: "r"(tmem_addr),
               "r"(vals[0]),"r"(vals[1]),"r"(vals[2]),"r"(vals[3]),
               "r"(vals[4]),"r"(vals[5]),"r"(vals[6]),"r"(vals[7]),
               "r"(vals[8]),"r"(vals[9]),"r"(vals[10]),"r"(vals[11]),
               "r"(vals[12]),"r"(vals[13]),"r"(vals[14]),"r"(vals[15]) : "memory");
    }
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    asm volatile("tcgen05.wait::st.sync.aligned;");
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t2));
#elif OP == 1
    // N writes WITH wait after each one
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        asm volatile("tcgen05.st.sync.aligned.16x64b.x16.b32 "
            "[%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
            :: "r"(tmem_addr),
               "r"(vals[0]),"r"(vals[1]),"r"(vals[2]),"r"(vals[3]),
               "r"(vals[4]),"r"(vals[5]),"r"(vals[6]),"r"(vals[7]),
               "r"(vals[8]),"r"(vals[9]),"r"(vals[10]),"r"(vals[11]),
               "r"(vals[12]),"r"(vals[13]),"r"(vals[14]),"r"(vals[15]) : "memory");
        asm volatile("tcgen05.wait::st.sync.aligned;");
    }
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t2));
#endif

    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem_addr));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;  // time without wait
        ((unsigned long long*)C)[1] = t2 - t1;  // wait drain time
    }
}
