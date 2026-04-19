// tcgen05.mma kind::f16 POWER microbench v2 - higher utilization version
// Bigger MMA shape (m=128 n=128 k=16) → 524288 FLOPs/inst (32x m=64 n=8)
// Single very-long inner loop (no per-outer commit overhead)
// SMEM-resident A/B with controllable pattern; no DRAM/L2 in steady state

#ifndef MMA_M
#define MMA_M 64
#endif
#ifndef MMA_N
#define MMA_N 8
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int iters, int pattern, int u2) {
    // SMEM: A (M=128 × K=16 fp16 = 4KB), B (K=16 × N=128 = 4KB)
    __shared__ __align__(1024) unsigned smem_A[2048];   // 8KB
    __shared__ __align__(1024) unsigned smem_B[2048];   // 8KB
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    // Pattern fill
    unsigned base;
    if      (pattern == 0) base = 0x00000000u;
    else if (pattern == 1) base = 0xFFFFFFFFu;
    else if (pattern == 2) base = 0x55555555u;
    else if (pattern == 3) base = 0xAAAAAAAAu;
    else if (pattern == 4) base = 0xDEADBEEFu;
    else if (pattern == 5) base = 0x3C003C00u;  // F16 +1.0 packed
    else                   base = 0x00000000u;

    if (threadIdx.x < 32) {
        for (int i = 0; i < 64; i++) {
            unsigned idx = threadIdx.x + i*32;
            unsigned w_a = (pattern == 4) ? (base ^ idx * 0xCAFEBABEu) : base;
            unsigned w_b = (pattern == 4) ? (base ^ idx * 0x13579BDFu) : base;
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

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    __syncthreads();
    unsigned tmem_addr = tmem_slot;

    // idesc encoding for m=128 n=128 k=16 kind::f16
    unsigned idesc = (1U << 4) | (1U << 7) | (1U << 10)
                   | (((unsigned)MMA_N >> 3) << 17)
                   | (((unsigned)MMA_M >> 4) << 24);

    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    // For m=128 n=128 k=16 K-major no swizzle:
    // A: 128×16 fp16 = 256 bytes/row × 128 rows = 32KB?? too big for 8KB smem.
    // Actually for m=128 k=16, A is 128 rows × 16 cols = 2048 elements × 2B = 4KB. Fits.
    // LBO = 32B (one row of 16 fp16); SBO = 128 × 32B / 8 = 512 (8-row swizzle group)
    // Use simpler: LBO=32 (16 fp16 = 32B per row), SBO=4096 (128 rows × 32B)
    unsigned long long LBO = 16, SBO = 128;  // proven for m=64 n=8 (existing test)
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned disable_lane[4] = {0,0,0,0};

    unsigned long long t0=0, t1=0;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (threadIdx.x == 0) {
        unsigned enable_d = 0;
        // Single long loop - no commits in middle
        for (int i = 0; i < iters; i++) {
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
        // commit + wait once at end
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

    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem_addr));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        long flops_per_inst = (long)MMA_M * MMA_N * 16 * 2;
        double tflops_per_cta = (double)iters * flops_per_inst / ((double)(t1-t0)/1.005e9) / 1e12;
        double tflops_total = tflops_per_cta * 148;
        printf("M=%d N=%d K=16 pattern=%d iters=%d cycles=%llu cy/MMA=%.2f per-CTA-TF=%.1f total-TF=%.1f\n",
               MMA_M, MMA_N, pattern, iters, t1-t0, (double)(t1-t0)/iters, tflops_per_cta, tflops_total);
    }
}
