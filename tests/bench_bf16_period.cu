// BF16 tcgen05.mma sign-period sweep at M=256 N=256 (2-CTA cluster)
// Period_size in N direction; non-sign bits held at random
// Args: -1 period_size

#define MMA_M 256
#define MMA_N 256
#define MMA_K 16

#define SIGN_MASK 0x80008000u   // bit 15 of each BF16 in packed unsigned

extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void kernel(float* A, float* B, float* C, int iters, int period_size, int u2) {
    __shared__ __align__(1024) unsigned smem_A[2048];
    __shared__ __align__(1024) unsigned smem_B[2048];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    // A: random per-block (M is split between 2 CTAs)
    if (threadIdx.x < 32) {
        for (int i = 0; i < 64; i++) {
            unsigned idx = threadIdx.x + i * 32;
            smem_A[idx] = 0xDEADBEEFu ^ (idx + blockIdx.x * 1024) * 0xCAFEBABEu;
        }
    }
    // B: each unsigned has 2 BF16. Each CTA fills its B SMEM (1024 words for N=256, K=16, half-N per CTA).
    // Wait: with 2-CTA M=256 N=256, B is shared across CTAs for the same K, N.
    // For 2-CTA cluster, each CTA owns its own B SMEM region; cluster-shared via descriptor.
    // For simplicity: same B fill per CTA (2 CTAs each fill their own B with same period pattern).
    // Each unsigned word: 2 BF16, indices k * (N/2) + n_pack
    int npacks = MMA_N / 2;  // 128 packs per K row
    int total_b_words = MMA_K * npacks;  // 16 * 128 = 2048
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < total_b_words; idx += 32) {
            int k = idx / npacks;
            int npack = idx % npacks;
            int n0 = npack * 2;     // first N in this word
            int n1 = n0 + 1;        // second N
            unsigned r = 0xDEADBEEFu ^ (idx + blockIdx.x * 1024) * 0x13579BDFu;
            r ^= r >> 16; r *= 0xCAFEBABEu;
            // Random non-sign bits for both BF16s
            unsigned word = r & ~SIGN_MASK;
            // Apply signs based on period
            int s0 = (n0 / period_size) & 1;
            int s1 = (n1 / period_size) & 1;
            if (s0) word |= (1u << 15);
            if (s1) word |= (1u << 31);
            // Force non-extreme exp to avoid Inf/NaN
            // BF16 exp at bits 7-14 (excluding sign). Force bit 14 = 0 (exp < 128, finite)
            word &= ~0x40004000u;  // clear bit 14 of each
            word |= 0x3E003E00u;   // set exp ~= 124 (small finite values)
            smem_B[idx] = word;
        }
    }
    if (threadIdx.x == 0) {
        tmem_slot = 0xFFFFFFFFu;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)));
    }
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");

    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    unsigned tmem_addr = tmem_slot;

    // idesc: bf16 = a_format=1 (BF16), b_format=1, c_format=2 (FP32)
    unsigned idesc = (1U << 4)                      // c_format = F32
                   | (1U << 7)                      // a_format = BF16
                   | (1U << 10)                     // b_format = BF16
                   | (((unsigned)MMA_N >> 3) << 17) // n_dim = 32
                   | (((unsigned)MMA_M >> 4) << 24); // m_dim = 16

    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16;
    unsigned long long SBO = 2 * (unsigned long long)(MMA_M / 2);  // per-CTA M
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned disable_lane[8] = {0,0,0,0,0,0,0,0};

    unsigned long long t0=0, t1=0;
    if (threadIdx.x == 0 && blockIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if ((blockIdx.x % 2) == 0 && threadIdx.x == 0) {
        unsigned enable_d = 0;
        for (int i = 0; i < iters; i++) {
            asm volatile(
                "{\n\t .reg .pred PRED;\n\t setp.ne.b32 PRED, %12, 0;\n\t"
                "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, "
                "{%4, %5, %6, %7, %8, %9, %10, %11}, PRED;\n\t}"
                :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                   "r"(disable_lane[0]), "r"(disable_lane[1]), "r"(disable_lane[2]), "r"(disable_lane[3]),
                   "r"(disable_lane[4]), "r"(disable_lane[5]), "r"(disable_lane[6]), "r"(disable_lane[7]),
                   "r"(enable_d)
                : "memory");
            enable_d = 1;
        }
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
        unsigned phase = 0;
        asm volatile(
            "{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase));
        if (blockIdx.x == 0)
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        printf("BF16 M=%d N=%d K=%d cta=2 period=%d cy/MMA=%.2f\n",
               MMA_M, MMA_N, MMA_K, period_size, (double)(t1-t0)/iters);
    }
}
