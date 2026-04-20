// 2-CTA tcgen05.mma POWER microbench
// Cluster (2,1): 2 CTAs paired for utcmma.cta_group::2
// Effective MMA shape: m=256 n=N k=K (m doubles for 2cta mode)
// Each CTA owns its half of A and B SMEM; leader issues utcmma.cta_group::2

#ifndef PRECISION
#define PRECISION 1   // 0=FP16, 1=BF16, 2=FP8E4M3, 3=FP8E5M2
#endif
#ifndef MMA_M
#define MMA_M 256   // 2cta mode supports m=128 or m=256
#endif
#ifndef MMA_N
#define MMA_N 256
#endif

#if PRECISION == 0
  #define KIND "f16"
  #define MMA_K 16
  #define A_FMT 0
  #define B_FMT 0
  #define POS_ONE_WORD 0x3C003C00u
#elif PRECISION == 1
  #define KIND "f16"
  #define MMA_K 16
  #define A_FMT 1
  #define B_FMT 1
  #define POS_ONE_WORD 0x3F803F80u
#elif PRECISION == 2
  #define KIND "f8f6f4"
  #define MMA_K 32
  #define A_FMT 0
  #define B_FMT 0
  #define POS_ONE_WORD 0x38383838u
#elif PRECISION == 3
  #define KIND "f8f6f4"
  #define MMA_K 32
  #define A_FMT 1
  #define B_FMT 1
  #define POS_ONE_WORD 0x3C3C3C3Cu
#endif

extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void kernel(float* A, float* B, float* C, int iters, int a_pat, int b_pat) {
    __shared__ __align__(1024) unsigned smem_A[2048];
    __shared__ __align__(1024) unsigned smem_B[2048];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    auto pat_word = [](int p) -> unsigned {
        if      (p == 0) return 0x00000000u;
        else if (p == 4) return 0xDEADBEEFu;
        else if (p == 5) return POS_ONE_WORD;
        else if (p == 2) return 0x55555555u;
        return 0x00000000u;
    };
    unsigned base_a = pat_word(a_pat);
    unsigned base_b = pat_word(b_pat);

    if (threadIdx.x < 32) {
        for (int i = 0; i < 64; i++) {
            unsigned idx = threadIdx.x + i*32;
            unsigned w_a = (a_pat == 4) ? (base_a ^ (idx + blockIdx.x*1024) * 0xCAFEBABEu) : base_a;
            unsigned w_b = (b_pat == 4) ? (base_b ^ (idx + blockIdx.x*1024) * 0x13579BDFu) : base_b;
            smem_A[idx] = w_a;
            smem_B[idx] = w_b;
        }
    }
    if (threadIdx.x == 0) {
        tmem_slot = 0xFFFFFFFFu;
        // FIX: commit.cta_group::2.mbarrier::arrive::one arrives ONCE only on leader's mbarrier
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)));
    }
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");

    // 2-CTA TMEM alloc: requires cta_group::2
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    unsigned tmem_addr = tmem_slot;

    // idesc: m_dim=16 for M=256 (2cta), n_dim=N/8
    unsigned idesc = (1U << 4)                            // c_format=F32
                   | (((unsigned)A_FMT) << 7)
                   | (((unsigned)B_FMT) << 10)
                   | (((unsigned)MMA_N >> 3) << 17)
                   | (((unsigned)MMA_M >> 4) << 24);

    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16;
    unsigned long long SBO = 2 * (unsigned long long)(MMA_M / 2);  // per-CTA chunk
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned disable_lane[8] = {0,0,0,0,0,0,0,0};

    unsigned long long t0=0, t1=0;
    if (threadIdx.x == 0 && blockIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Only LEADER CTA (blockIdx.x == 0 in cluster) issues utcmma.cta_group::2
    if ((blockIdx.x % 2) == 0 && threadIdx.x == 0) {  // FIX: cluster-relative leader (every even block)
        unsigned enable_d = 0;
        for (int i = 0; i < iters; i++) {
            // 2-CTA MMA has 8-element disable_lane mask
            asm volatile(
                "{\n\t .reg .pred PRED;\n\t"
                "setp.ne.b32 PRED, %12, 0;\n\t"
                "tcgen05.mma.cta_group::2.kind::" KIND " [%0], %1, %2, %3, "
                "{%4, %5, %6, %7, %8, %9, %10, %11}, PRED;\n\t"
                "}"
                :
                : "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                  "r"(disable_lane[0]), "r"(disable_lane[1]), "r"(disable_lane[2]), "r"(disable_lane[3]),
                  "r"(disable_lane[4]), "r"(disable_lane[5]), "r"(disable_lane[6]), "r"(disable_lane[7]),
                  "r"(enable_d)
                : "memory");
            enable_d = 1;
        }
        // 2-CTA commit
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
    }
    // FIX: only LEADER waits on mbarrier; non-leader just sync via cluster barrier
    if ((blockIdx.x % 2) == 0 && threadIdx.x == 0) {  // FIX: cluster-relative leader (every even block)
        unsigned phase = 0;
        asm volatile(
            "{\n\t .reg .pred P;\n\t"
            "WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase));
    }
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    if (threadIdx.x == 0 && blockIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");

    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        long flops_per_inst = (long)MMA_M * MMA_N * MMA_K * 2;
        // 2-CTA mode: only 74 SM pairs active (148/2)
        double tflops_per_pair = (double)iters * flops_per_inst / ((double)(t1-t0)/1.005e9) / 1e12;
        double tflops_total = tflops_per_pair * 74;
        printf("2CTA PRECISION=%d M=%d N=%d K=%d a=%d b=%d iters=%d cy/MMA=%.2f total-TF=%.1f\n",
               PRECISION, MMA_M, MMA_N, MMA_K, a_pat, b_pat, iters,
               (double)(t1-t0)/iters, tflops_total);
    }
}
