// tcgen05.mma kind::mxf4nvf4.block_scale.block16 POWER microbench
// Properly initializes SFA and SFB TMEM regions to UE4M3 1.0 (byte 0x38)
// via tcgen05.st (excessively, to ensure all reads land on 1.0)
// K_SIZE: 0 = K64 (dense), 1 = K96 (NVFP4 ULTRA)

#ifndef MMA_M
#define MMA_M 128
#endif
#ifndef MMA_N
#define MMA_N 128
#endif
#ifndef K_SIZE
#define K_SIZE 0   // 0=K64, 1=K96
#endif
#if K_SIZE == 0
#define MMA_K 64
#else
#define MMA_K 96
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int iters, int a_pat, int b_pat) {
    __shared__ __align__(1024) unsigned smem_A[2048];
    __shared__ __align__(1024) unsigned smem_B[2048];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    auto pat_word = [](int p) -> unsigned {
        if      (p == 0) return 0x00000000u;
        else if (p == 4) return 0xDEADBEEFu;
        else if (p == 5) return 0x22222222u;     // FP4 +1.0 (0x2 packed)
        else if (p == 2) return 0x55555555u;
        return 0x00000000u;
    };
    unsigned base_a = pat_word(a_pat);
    unsigned base_b = pat_word(b_pat);

    if (threadIdx.x < 32) {
        for (int i = 0; i < 64; i++) {
            unsigned idx = threadIdx.x + i*32;
            unsigned w_a = (a_pat == 4) ? (base_a ^ idx * 0xCAFEBABEu) : base_a;
            unsigned w_b = (b_pat == 4) ? (base_b ^ idx * 0x13579BDFu) : base_b;
            smem_A[idx] = w_a;
            smem_B[idx] = w_b;
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
    unsigned tsfa_addr = tmem_addr + 128;
    unsigned tsfb_addr = tmem_addr + 256;

    // Fill ENTIRE 512 cols of TMEM with 0x38383838 (UE4M3 1.0 packed 4×)
    // Each tcgen05.st.sync.aligned.32x32b.x4.b32 writes 4 cols (128 rows × 4 cols = 512 elements per warp inst)
    // For 512 cols total: 32 lanes × 4 cols/inst × N insts. With N=4 → 32*4*4 = 512 cols. Each lane does 4 sts of 4-cols each.
    {
        unsigned one_pack = 0x38383838u;
        for (int chunk = 0; chunk < 4; chunk++) {
            unsigned col_base = chunk * 128 + (threadIdx.x * 4);  // each lane does 4 cols, 32 lanes per chunk = 128 cols
            unsigned addr = tmem_addr + col_base;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};\n"
                :: "r"(addr), "r"(one_pack), "r"(one_pack), "r"(one_pack), "r"(one_pack));
        }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;");
    __syncthreads();

    // idesc encoding for InstrDescriptorBlockScaled (mxf4nvf4)
    unsigned idesc = (5U << 7)                      // a_format = E2M1
                   | (5U << 10)                     // b_format = E2M1
                   | (((unsigned)MMA_N >> 3) << 17) // n_dim
                   | (((unsigned)MMA_M >> 4) << 24) // m_dim
                   | (((unsigned)K_SIZE) << 31);    // k_size: 0=K64, 1=K96

    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    // SBO = 2*MMA_M (empirical from existing tests: m=64→128, m=128→256, m=256→512)
    unsigned long long LBO = 16;
    unsigned long long SBO = 2 * (unsigned long long)MMA_M;
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);

    unsigned long long t0=0, t1=0;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (threadIdx.x == 0) {
        unsigned scaleC = 0;
        for (int i = 0; i < iters; i++) {
            asm volatile(
                "{\n\t .reg .pred PRED;\n\t"
                "setp.ne.b32 PRED, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], PRED;\n\t"
                "}"
                :
                : "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                  "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr)
                : "memory");
            scaleC = 1;
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
        unsigned phase = 0;
        asm volatile(
            "{\n\t .reg .pred P;\n\t"
            "WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();

    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        long flops_per_inst = (long)MMA_M * MMA_N * MMA_K * 2;
        double tflops_per_cta = (double)iters * flops_per_inst / ((double)(t1-t0)/1.005e9) / 1e12;
        double tflops_total = tflops_per_cta * 148;
        printf("NVFP4 K_SIZE=%d (k=%d) m=%d n=%d a_pat=%d b_pat=%d iters=%d cy/MMA=%.2f total-TF=%.1f\n",
               K_SIZE, MMA_K, MMA_M, MMA_N, a_pat, b_pat, iters, (double)(t1-t0)/iters, tflops_total);
    }
}
