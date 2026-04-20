// MXFP8 e4m3 with block_scale.block32 sign-period sweep
// Same FP8 layout as bench_fp8_period.cu but with SF tensor (UE4M3 1.0)

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
    int npacks = MMA_N / 4;
    int total_b_words = MMA_K * npacks;
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < total_b_words; idx += 32) {
            int k = idx / npacks;
            int npack = idx % npacks;
            int n_base = npack * 4;
            unsigned r = 0xDEADBEEFu ^ (idx + blockIdx.x * 1024) * 0x13579BDFu;
            r ^= r >> 16; r *= 0xCAFEBABEu;
            unsigned word = 0;
            for (int p = 0; p < 4; p++) {
                int n = n_base + p;
                int sign = (n / period_size) & 1;
                unsigned char fp8_bits = (r >> (p * 8)) & 0x7F;
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
    unsigned tsfa_addr = tmem_addr + 128;
    unsigned tsfb_addr = tmem_addr + 256;

    // Init SF tensor to UE8M0 1.0 (byte 0x7F) - REQUIRED for kind::mxf8f6f4
    {
        unsigned one_pack = 0x7F7F7F7Fu;
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

    // MXFP8 e4m3: kind::mxf8f6f4 with block_scale.block32
    // No c_format in block-scaled idesc (bit 4 = b_sf_id, leave 0)
    // bit 23 = scale_format = 1 = UE8M0
    unsigned idesc = (0U << 7)                       // a_format = E4M3
                   | (0U << 10)                      // b_format = E4M3
                   | (((unsigned)MMA_N >> 3) << 17)
                   | (1U << 23)                      // scale_format = UE8M0
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

    unsigned long long t0=0, t1=0;
    if (threadIdx.x == 0 && blockIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if ((blockIdx.x % 2) == 0 && threadIdx.x == 0) {
        unsigned scaleC = 0;
        for (int i = 0; i < iters; i++) {
            asm volatile(
                "{\n\t .reg .pred PRED;\n\t setp.ne.b32 PRED, %4, 0;\n\t"
                "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], PRED;\n\t}"
                :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                   "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr) : "memory");
            scaleC = 1;
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
        printf("MXFP8 M=%d N=%d K=%d cta=2 period=%d cy/MMA=%.2f\n",
               MMA_M, MMA_N, MMA_K, period_size, (double)(t1-t0)/iters);
    }
}
