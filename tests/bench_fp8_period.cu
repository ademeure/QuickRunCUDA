// FP8 e4m3 sign-period sweep at M=N=256 (2-CTA cluster)
// FP8 e4m3: bit 7=sign, bits 6-3=exp(4), bits 2-0=mantissa(3)
// 4 FP8 per 32-bit word; sign mask = 0x80808080

#define MMA_M 256
#define MMA_N 256
#define MMA_K 32
#define SIGN_MASK 0x80808080u

extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void kernel(float* A, float* B, float* C, int iters, int period_size, int u2) {
    __shared__ __align__(1024) unsigned smem_A[2048];
    __shared__ __align__(1024) unsigned smem_B[2048];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    if (threadIdx.x < 32) {
        for (int i = 0; i < 64; i++) {
            unsigned idx = threadIdx.x + i * 32;
            smem_A[idx] = 0xDEADBEEFu ^ (idx + blockIdx.x * 1024) * 0xCAFEBABEu;
        }
    }
    // B: 4 FP8 per word. K=32, N=256 (per CTA half = 128 N).
    // Total FP8 per CTA = K * N = 32 * 256 = 8192 FP8 = 2048 unsigned
    int npacks = MMA_N / 4;  // 64 packs per K row
    int total_b_words = MMA_K * npacks;  // 32*64 = 2048
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < total_b_words; idx += 32) {
            int k = idx / npacks;
            int npack = idx % npacks;
            int n_base = npack * 4;
            unsigned r = 0xDEADBEEFu ^ (idx + blockIdx.x * 1024) * 0x13579BDFu;
            r ^= r >> 16; r *= 0xCAFEBABEu;
            // For each of 4 FP8 in word, build value with controlled sign
            unsigned word = 0;
            for (int p = 0; p < 4; p++) {
                int n = n_base + p;
                int sign = (n / period_size) & 1;
                // Random non-sign bits (7 bits)
                unsigned char fp8_bits = (r >> (p * 8)) & 0x7F;
                // Force exp into normal range (avoid Inf/NaN)
                unsigned char exp = (fp8_bits >> 3) & 0xF;
                if (exp == 0) exp = 1;
                if (exp == 15) exp = 14;
                fp8_bits = (fp8_bits & 0x07) | (exp << 3);
                if (sign) fp8_bits |= 0x80;
                word |= ((unsigned)fp8_bits << (p * 8));
            }
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

    // FP8 e4m3: a_format = 0 (FP8 e4m3), b_format = 0
    unsigned idesc = (1U << 4)                       // c_format = F32
                   | (0U << 7)                       // a_format = E4M3
                   | (0U << 10)                      // b_format = E4M3
                   | (((unsigned)MMA_N >> 3) << 17)
                   | (((unsigned)MMA_M >> 4) << 24);

    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16;
    unsigned long long SBO = 2 * (unsigned long long)(MMA_M / 2);
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
                "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, "
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
        printf("FP8 M=%d N=%d K=%d cta=2 period=%d cy/MMA=%.2f\n",
               MMA_M, MMA_N, MMA_K, period_size, (double)(t1-t0)/iters);
    }
}
