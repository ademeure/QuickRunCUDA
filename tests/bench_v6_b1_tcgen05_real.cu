// tcgen05.mma.kind::f16 with proper descriptors for real MMA
// Matrix shape: m=128, n=128, k=16 → 128×16 A in smem, 16×128 B in smem
// Accumulator: 128×128 in TMEM (need 128 cols × 128 lanes)

#ifndef ITERS
#define ITERS 1000
#endif

// Make descriptor for smem matrix
__device__ inline unsigned long long make_desc(unsigned smem_addr_bytes, unsigned lbo, unsigned sbo, unsigned swizzle) {
    unsigned long long d = 0;
    d |= ((unsigned long long)(smem_addr_bytes >> 4) & 0x3FFF);  // 14 bits
    d |= ((unsigned long long)(lbo >> 4) & 0x3FFF) << 16;
    d |= ((unsigned long long)(sbo >> 4) & 0x3FFF) << 32;
    d |= (unsigned long long)(swizzle & 0x3) << 52;
    return d;
}

extern "C" __global__ __launch_bounds__(128, 1)
void kernel(float* A, float* B, float* C, int u0, int seed, int u2) {
    __shared__ __align__(1024) unsigned smem_A[128 * 16 / 2];  // 128×16 half = 2 KB
    __shared__ __align__(1024) unsigned smem_B[16 * 128 / 2];  // 16×128 half = 2 KB
    __shared__ __align__(4) unsigned int tmem_slot;

    // Init smem
    if (threadIdx.x < 128) {
        for (int i = 0; i < 8; i++) {
            unsigned idx = threadIdx.x + i * 128;
            smem_A[idx] = 0x3C003C00;
            smem_B[idx] = 0x3C003C00;
        }
    }
    if (threadIdx.x == 0) tmem_slot = 0xFFFFFFFF;
    __syncthreads();

    // Alloc TMEM (1 warpgroup of 128 threads needed for alloc)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    __syncthreads();
    unsigned tmem_addr = tmem_slot;
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");

    // Build descriptors (minimal - swizzle=0, LBO/SBO = 0)
    unsigned a_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long a_desc = make_desc(a_addr, 16, 16, 0);  // FP16 matrix
    unsigned long long b_desc = make_desc(b_addr, 16, 16, 0);
    // idesc format: bits 0-4 = m (128/8 - 1 = 15), bits 6-10 = n (128/8 - 1 = 15), etc.
    // For m=128,n=128,k=16,FP16: simplest idesc has all default fields
    unsigned idesc = 0;
    
    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    __syncthreads();

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile(
            "{.reg .pred P;\n\t"
            "setp.ne.b32 P, 1, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, P;}"
            :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc) : "memory");
    }
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;");

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    __syncthreads();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" :: "r"(tmem_addr));
    
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
    }
}
