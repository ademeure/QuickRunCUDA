// tcgen05.mma kind::f16 with NO swizzle (layout_type=0)
// Assumes linear smem matrices, descriptor encodes byte strides directly.

#ifndef ITERS
#define ITERS 10
#endif
#ifndef MMA_M
#define MMA_M 64
#endif
#ifndef MMA_N
#define MMA_N 8
#endif
#ifndef LAYOUT
#define LAYOUT 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int u0, int seed, int u2) {
    __shared__ __align__(1024) unsigned smem_A[2048];  // 8 KB max
    __shared__ __align__(1024) unsigned smem_B[2048];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    if (threadIdx.x < 32) {
        for (int i = 0; i < 64; i++) smem_A[threadIdx.x + i*32] = 0x3F803F80;
        for (int i = 0; i < 64; i++) smem_B[threadIdx.x + i*32] = 0x3F803F80;
    }
    if (threadIdx.x == 0) {
        tmem_slot = 0xFFFFFFFF;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)));
    }
    __syncthreads();

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    __syncthreads();
    unsigned tmem_addr = tmem_slot;

    unsigned idesc = (1U << 4)
                   | (1U << 7)
                   | (1U << 10)
                   | (((unsigned)MMA_N >> 3) << 17)
                   | (((unsigned)MMA_M >> 4) << 24);

    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    // No-swizzle: LBO = stride within atom along leading dim, SBO = stride between atoms in stride dim
    // For K-major FP16 atom (8x8 elements = 8 rows × 16 bytes per row):
    // LBO = 16B (one row of 8 FP16 elements)
    // SBO = 8 rows × 16 B = 128 B
    unsigned long long LBO = 16;
    unsigned long long SBO = 128;
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32)
                              | ((unsigned long long)LAYOUT << 61);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32)
                              | ((unsigned long long)LAYOUT << 61);

    unsigned disable_lane[4] = {0, 0, 0, 0};

    unsigned long long t0 = 0, t1 = 0;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (threadIdx.x == 0) {
        unsigned enable_d = 0;
        for (int i = 0; i < ITERS; i++) {
            asm volatile(
                "{\n\t .reg .pred PRED; \n\t"
                "setp.ne.b32 PRED, %8, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%4, %5, %6, %7}, PRED;\n\t"
                "}"
                :
                : "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                  "r"(disable_lane[0]), "r"(disable_lane[1]), "r"(disable_lane[2]), "r"(disable_lane[3]),
                  "r"(enable_d)
                : "memory");
            enable_d = 1;
        }

        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");

        unsigned phase = 0;
        asm volatile(
            "{\n\t .reg .pred P;\n\t"
            "WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@P bra DONE;\n\t"
            "bra WAIT;\n\t"
            "DONE:\n\t"
            "}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase));

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();

    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" :: "r"(tmem_addr));

    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = (float)idesc;
        ((unsigned long long*)C)[3] = (float)tmem_addr;
        printf("ITERS=%d cycles=%llu cy/iter=%.2f idesc=0x%x a_desc=0x%llx tmem=0x%x\n",
               ITERS, t1-t0, (double)(t1-t0)/ITERS, idesc, a_desc, tmem_addr);
    }
}
