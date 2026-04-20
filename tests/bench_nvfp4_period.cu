// NVFP4 sign-pattern sweep: vary alternation period in N direction
// arg1 = period_size (1, 2, 3, 4, 6, 8, ..., up to N)
// arg2 = unused (u2)
// Compile-time: K_SIZE, MMA_N, CTA_GROUP

#ifndef MMA_M
#define MMA_M 128
#endif
#ifndef MMA_N
#define MMA_N 128
#endif
#ifndef K_SIZE
#define K_SIZE 0
#endif
#ifndef CTA_GROUP
#define CTA_GROUP 1
#endif
#if K_SIZE == 0
#define MMA_K 64
#else
#define MMA_K 96
#endif

// Total B SMEM size in unsigned words (8 FP4 per word)
#define B_SMEM_WORDS (MMA_K * MMA_N / 8)
#define A_SMEM_WORDS (MMA_K * MMA_M / 8)

extern "C" __global__ __launch_bounds__(32, 1)
#if CTA_GROUP == 2
__cluster_dims__(2, 1, 1)
#endif
void kernel(float* A, float* B, float* C, int iters, int period_size, int u2) {
    __shared__ __align__(1024) unsigned smem_A[3072];
    __shared__ __align__(1024) unsigned smem_B[3072];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    int smem_a_size = MMA_K * MMA_M / 8;  // packed FP4
    int smem_b_size = B_SMEM_WORDS;

    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_a_size; idx += 32) {
            smem_A[idx] = 0xDEADBEEFu ^ idx * 0xCAFEBABEu;
        }
    }
    // B: signs alternate with period in N direction
    // Each unsigned has 8 FP4 = 8 N positions
    // Within word at index (k * (MMA_N/8) + npack):
    //   FP4 at position p (0..7) is at N-coord = npack*8 + p
    //   Sign = ((N-coord / period_size) % 2) → 0 or 1
    int n_per_word = 8;
    int npacks = MMA_N / n_per_word;  // 1, 2, 4, 8, 12, 16, 24, 28, 32 for N=8..256
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_b_size; idx += 32) {
            int k = idx / npacks;
            int npack = idx % npacks;
            int n_base = npack * n_per_word;
            unsigned r = 0xDEADBEEFu ^ idx * 0x13579BDFu;
            r ^= r >> 16; r *= 0xCAFEBABEu;
            // Build word with sign bits according to period
            unsigned val = 0;
            for (int p = 0; p < n_per_word; p++) {
                int n = n_base + p;
                unsigned fp4 = (r >> (p*4)) & 0x7;  // random bits 0-2 (no sign)
                // Set sign based on period
                int sign = (period_size > 0) ? ((n / period_size) & 1) : 0;
                fp4 |= (sign << 3);
                val |= (fp4 << (p * 4));
            }
            smem_B[idx] = val;
        }
    }
    if (threadIdx.x == 0) {
        tmem_slot = 0xFFFFFFFFu;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)));
    }
    __syncthreads();

#if CTA_GROUP == 1
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
#else
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
#endif
    __syncthreads();
    unsigned tmem_addr = tmem_slot;
    unsigned tsfa_addr = tmem_addr + 128;
    unsigned tsfb_addr = tmem_addr + 256;

    {
        unsigned one_pack = 0x38383838u;
        for (int chunk = 0; chunk < 4; chunk++) {
            unsigned col_base = chunk * 128 + (threadIdx.x * 4);
            unsigned addr = tmem_addr + col_base;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};\n"
                :: "r"(addr), "r"(one_pack), "r"(one_pack), "r"(one_pack), "r"(one_pack));
        }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;");
    __syncthreads();

    unsigned idesc = (5U << 7) | (5U << 10)
                   | (((unsigned)MMA_N >> 3) << 17)
                   | (((unsigned)MMA_M >> 4) << 24)
                   | (((unsigned)K_SIZE) << 31);

    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16;
    unsigned long long SBO = 2 * (unsigned long long)MMA_M;
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);

    unsigned long long t0=0, t1=0;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (threadIdx.x == 0) {
        unsigned scaleC = 0;
        for (int i = 0; i < iters; i++) {
#if CTA_GROUP == 1
            asm volatile(
                "{\n\t .reg .pred P;\n\t setp.ne.b32 P, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], P;\n\t}"
                :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                   "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr) : "memory");
#else
            asm volatile(
                "{\n\t .reg .pred P;\n\t setp.ne.b32 P, %4, 0;\n\t"
                "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], P;\n\t}"
                :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                   "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr) : "memory");
#endif
            scaleC = 1;
        }
#if CTA_GROUP == 1
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
#else
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
#endif
        unsigned phase = 0;
        asm volatile(
            "{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();
#if CTA_GROUP == 1
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
#else
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
#endif

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        printf("NVFP4 K=%d N=%d cta=%d period=%d cy/MMA=%.2f\n",
               MMA_K, MMA_N, CTA_GROUP, period_size, (double)(t1-t0)/iters);
    }
}
