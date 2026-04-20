// Test: does dedup state SURVIVE between MMAs when B comes from different SMEM region?
// Setup: 2 SMEM B regions with DIFFERENT data. Alternate between them per MMA iteration.
// Compare power to single-region runs.

#define MMA_M 128
#define MMA_N 128
#define MMA_K 16

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int iters, int mode, int verify) {
    __shared__ __align__(1024) unsigned smem_A[2048];
    __shared__ __align__(1024) unsigned smem_B0[2048];  // Region 0
    __shared__ __align__(1024) unsigned smem_B1[2048];  // Region 1
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    // A always random
    if (threadIdx.x < 32) {
        for (int i = 0; i < 64; i++) {
            unsigned idx = threadIdx.x + i*32;
            smem_A[idx] = 0xDEADBEEFu ^ idx * 0xCAFEBABEu;
        }
    }

    // B Region 0: based on mode
    // B Region 1: based on mode (might be same or different)
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < 1024; idx += 32) {
            unsigned r0 = 0xDEADBEEFu ^ idx * 0x13579BDFu;
            unsigned r1 = 0xC0FFEE13u ^ idx * 0x9E3779B1u;  // Different random
            unsigned w0, w1;
            if (mode == 0) {
                // Both regions = same random
                w0 = r0;
                w1 = r0;
            } else if (mode == 1) {
                // Both = same const +1.0
                w0 = w1 = 0x3F803F80u;
            } else if (mode == 2) {
                // Region 0 = random, Region 1 = different random
                w0 = r0;
                w1 = r1;
            } else if (mode == 3) {
                // Region 0 = const, Region 1 = random
                w0 = 0x3F803F80u;
                w1 = r1;
            } else {
                w0 = r0;
                w1 = r0;
            }
            smem_B0[idx] = w0;
            smem_B1[idx] = w1;
        }
    }

    if (threadIdx.x == 0) {
        tmem_slot = 0xFFFFFFFFu;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)));
    }
    __syncthreads();

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    __syncthreads();
    unsigned tmem_addr = tmem_slot;

    unsigned idesc = (1U << 4) | (1U << 7) | (1U << 10)
                   | (((unsigned)MMA_N >> 3) << 17)
                   | (((unsigned)MMA_M >> 4) << 24);
    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b0_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B0);
    unsigned b1_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B1);
    unsigned long long LBO = 16, SBO = 256;
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b0_desc = desc_encode(b0_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b1_desc = desc_encode(b1_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned disable_lane[4] = {0,0,0,0};

    unsigned long long t0=0, t1=0;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (threadIdx.x == 0) {
        unsigned enable_d = 0;
        for (int i = 0; i < iters; i++) {
            unsigned long long b_desc;
            if (verify == 1) {
                // Alternate between regions
                b_desc = (i & 1) ? b1_desc : b0_desc;
            } else {
                // Always region 0
                b_desc = b0_desc;
            }
            asm volatile(
                "{\n\t .reg .pred PRED;\n\t setp.ne.b32 PRED, %8, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%4, %5, %6, %7}, PRED;\n\t}"
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
            "{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        printf("DescSwap mode=%d alt=%d iters=%d cy/MMA=%.2f\n",
               mode, verify, iters, (double)(t1-t0)/iters);
    }
}
