// tcgen05.mma POWER microbench: data pattern in SMEM ONLY (no DRAM/L2 in inner loop)
// Initialize A and B SMEM tiles ONCE, then issue many tcgen05.mma referencing the
// SAME SMEM addresses. Power profile reflects ONLY the multiplier consuming that
// constant SMEM data — DRAM/L2 contribution to power is zero in steady state.
//
// PATTERN selection via arg2 (seed):
//   0=zero(0x00) 1=ones(0xff) 2=0x55 3=0xaa 4=random 5=+1.0(F16=0x3C00→0x3C003C00)
//
// Defaults: kind::f16 m=64 n=8 k=16 (works with no swizzle, smallest single utcmma).
// Inner-loop ITERS=10000 → ~100k MMAs per CTA — long enough for dmon power capture.

#ifndef ITERS
#define ITERS 10000
#endif
#ifndef MMA_M
#define MMA_M 64
#endif
#ifndef MMA_N
#define MMA_N 8
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int outer_iters, int pattern, int u2) {
    __shared__ __align__(1024) unsigned smem_A[2048];
    __shared__ __align__(1024) unsigned smem_B[2048];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    // Fill SMEM with chosen pattern (each unsigned = 4 bytes = 2 FP16 values)
    unsigned fill_word;
    if (pattern == 0)      fill_word = 0x00000000u;     // zero
    else if (pattern == 1) fill_word = 0xFFFFFFFFu;     // 0xff = -inf in F16
    else if (pattern == 2) fill_word = 0x55555555u;     // alternating bits low
    else if (pattern == 3) fill_word = 0xAAAAAAAAu;     // alternating bits high
    else if (pattern == 4) fill_word = 0x12345678u + threadIdx.x * 0xDEADBEEFu;  // pseudo-random
    else if (pattern == 5) fill_word = 0x3C003C00u;     // F16 +1.0 / +1.0
    else                   fill_word = 0x00000000u;

    if (threadIdx.x < 32) {
        for (int i = 0; i < 64; i++) {
            unsigned idx = threadIdx.x + i*32;
            // For pattern==4 (random): different word per index for true randomness
            unsigned w = (pattern == 4) ? (fill_word ^ (idx * 0xCAFEBABEu)) : fill_word;
            smem_A[idx] = w;
            smem_B[idx] = w ^ (pattern == 4 ? 0x13579BDFu : 0u);  // B differs from A only in random mode
        }
    }
    if (threadIdx.x == 0) {
        tmem_slot = 0xFFFFFFFFu;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)));
    }
    __syncthreads();

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;"
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
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16, SBO = 128;
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned disable_lane[4] = {0,0,0,0};

    unsigned long long t0=0, t1=0;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (threadIdx.x == 0) {
        unsigned enable_d = 0;
        // OUTER loop drives the long sustained workload (for dmon); INNER is the
        // intrinsic burst of MMAs between commit+wait (keeps the issue queue full).
        // Total MMAs = outer_iters * ITERS (default 10000 outer * ITERS inner)
        for (int o = 0; o < outer_iters; o++) {
            #pragma unroll 1
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
            // commit+wait once per outer iter
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
            unsigned phase = o & 1;
            asm volatile(
                "{\n\t .reg .pred P;\n\t"
                "WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
                :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase));
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();

    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" :: "r"(tmem_addr));

    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = (float)idesc;
        ((unsigned*)C)[3] = pattern;
    }
}
