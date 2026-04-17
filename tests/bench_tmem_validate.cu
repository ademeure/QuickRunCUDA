// Validate TMEM write actually completes. Write known patterns, read back, verify.
// Multiple variants: with/without wait, ordering tests.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
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

#if OP == 0
    // Write known pattern, read back
    unsigned vals[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) vals[i] = (threadIdx.x << 16) | (i + 1);

    asm volatile("tcgen05.st.sync.aligned.16x64b.x16.b32 "
        "[%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
        :: "r"(tmem_addr),
           "r"(vals[0]),"r"(vals[1]),"r"(vals[2]),"r"(vals[3]),
           "r"(vals[4]),"r"(vals[5]),"r"(vals[6]),"r"(vals[7]),
           "r"(vals[8]),"r"(vals[9]),"r"(vals[10]),"r"(vals[11]),
           "r"(vals[12]),"r"(vals[13]),"r"(vals[14]),"r"(vals[15]) : "memory");

    asm volatile("tcgen05.wait::st.sync.aligned;");

    unsigned r[16];
    asm volatile("tcgen05.ld.sync.aligned.16x64b.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
          "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
          "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
          "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
        : "r"(tmem_addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");

    // Each thread checks its own values
    unsigned mismatches = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        if (r[i] != vals[i]) mismatches++;
    }
    // Output: total mismatches across all threads
    if (threadIdx.x == 0) {
        ((unsigned*)C)[0] = mismatches;
        ((unsigned*)C)[1] = vals[0];
        ((unsigned*)C)[2] = r[0];
    }
    if (mismatches > 0) C[blockIdx.x * 4 + threadIdx.x % 4 + 4] = (mismatches << 16) | threadIdx.x;
#endif
    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem_addr));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}
