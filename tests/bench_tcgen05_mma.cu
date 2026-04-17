// Minimal tcgen05.mma attempt. Alloc TMEM, run MMA with zero descriptors.
// May runtime-fail with illegal instruction if descriptors don't resolve.

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(4) unsigned int tmem_slot;
    __shared__ __align__(16) unsigned int smem_buf[512];
    if (threadIdx.x == 0) tmem_slot = 0xFFFFFFFF;
    __syncthreads();

    // Alloc 128 cols of tensor memory
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    __syncthreads();
    unsigned tmem_addr = tmem_slot;

    // Fake descriptors pointing at smem_buf (won't produce meaningful HMMA, but tests runtime path)
    unsigned long long a_desc = 0;
    unsigned long long b_desc = 0;
    unsigned int idesc = 0;

    unsigned long long t0 = 0, t1 = 0;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Use kind::f16 (simplest, Hopper compatible)
    // PTX: tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, P
    asm volatile(
        "{.reg .pred P;\n\t"
        "setp.ne.b32 P, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, P;}"
        :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc) : "memory");

    // commit + wait
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;");

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    // ld result from tmem
    unsigned r;
    asm volatile("tcgen05.ld.sync.aligned.16x64b.x1.b32 {%0}, [%1];"
        : "=r"(r) : "r"(tmem_addr));

    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" :: "r"(tmem_addr));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");

    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned int*)C)[2] = r;
        ((unsigned int*)C)[3] = tmem_addr;
    }
}
