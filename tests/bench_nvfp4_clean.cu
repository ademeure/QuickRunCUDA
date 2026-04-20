// Clean NVFP4 sign-bit power kernel - K=64/96, fixed SF init that won't crash
// Run with -1 sign_mode (0=random, 1=all-pos, 2=all-neg, 3=K-unif/N, 4=single/K, 31=alt)
// SF always = UE4M3 1.0 (byte 0x38) for stability

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

#define SIGN_MASK 0x88888888u

extern "C" __global__ __launch_bounds__(32, 1)
#if CTA_GROUP == 2
__cluster_dims__(2, 1, 1)
#endif
void kernel(float* A, float* B, float* C, int iters, int sign_mode, int u2) {
    __shared__ __align__(1024) unsigned smem_A[3072];
    __shared__ __align__(1024) unsigned smem_B[3072];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    int smem_size = MMA_K * 16;

    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            smem_A[idx] = 0xDEADBEEFu ^ idx * 0xCAFEBABEu;
        }
    }
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            int k = idx / 16;
            int npack = idx % 16;
            int n_base = npack * 8;
            unsigned r = 0xDEADBEEFu ^ idx * 0x13579BDFu;
            unsigned sign_set = 0;
            switch (sign_mode) {
                case 0: sign_set = (r >> 3) & 0xFF; break;
                case 1: sign_set = 0x00; break;
                case 2: sign_set = 0xFF; break;
                case 31: sign_set = 0xAA; break;
                case 3: { unsigned r0 = 0xDEADBEEFu ^ npack * 0x13579BDFu;
                          sign_set = (r0 >> 3) & 0xFF; break; }
                case 4: { unsigned r0 = 0xDEADBEEFu ^ k * 0x13579BDFu;
                          sign_set = ((r0>>15)&1) ? 0xFF : 0x00; break; }
                default: sign_set = (r >> 3) & 0xFF;
            }
            unsigned sign_bits = 0;
            for (int p = 0; p < 8; p++) if ((sign_set>>p)&1) sign_bits |= (1u<<(4*p+3));
            smem_B[idx] = (r & ~SIGN_MASK) | sign_bits;
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

    // SF init: u2 controls (0=1.0, 1=zero, 2=patterned 0xAA)
    {
        unsigned one_pack = 0x38383838u;
        if (u2 == 1) one_pack = 0x00000000u;
        else if (u2 == 2) one_pack = 0x39393939u; // close-to-1.0 alt pattern
        else if (u2 == 3) one_pack = 0x36363636u; // 0.5
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
        printf("NVFP4 K=%d M=%d N=%d cta=%d sign=%d cy/MMA=%.2f\n",
               MMA_K, MMA_M, MMA_N, CTA_GROUP, sign_mode, (double)(t1-t0)/iters);
    }
}
