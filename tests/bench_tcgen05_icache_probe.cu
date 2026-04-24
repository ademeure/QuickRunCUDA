// bench_tcgen05_icache_probe.cu
// I-cache hypothesis probe for §21 sustained-load tcgen05.mma cliff.
//
// Catalog claim (B300_PIPE_CATALOG.md L7797+):
//   ITERS  cy/MMA  TFLOPS  %peak   (single warp, sustained tcgen05.mma FP8)
//    5K    128.05  4654    100%
//   30K    128.01  4655    100% (cliff edge)
//   50K    305.90  1949    42%
//  100K    394.16  1512    32%
// Catalog mechanism: "dispatch bubbles ... possibly hardware running-average power
// tracking" or "tcgen05 internal queue/scheduler limits".
//
// HYPOTHESIS (user 2026-04-24): cliff is INSTRUCTION-CACHE thrashing.
//   - tcgen05.mma SASS (UTCQMMA / UTCOMMA / UTCHMMA) ~14 bytes
//   - 30K MMAs fully unrolled = ~420 KB code, far exceeds I-cache
//   - With #pragma unroll 1, body = 1 inst => I-cache footprint ~constant
//   - => unroll 1 should ELIMINATE the cliff entirely (provable)
//
// CONFIGURATION:
//   -H "#define MMA_KIND N"   shape: 0=f16(M64N8K16,~44cy), 1=f8(M128N128K32,~128cy),
//                                    2=mxf4(M128N256K64,~128cy)
//   -H "#define UNROLL N"     0=#pragma unroll 1; -1=no pragma; N>0=#pragma unroll N
//   -0 ITERS                  loop count
//
// Anti-DCE: clock64 + commit/wait + mbarrier ensure execution.
// Single CTA, 1 warp (matches catalog "from one warp").

#ifndef MMA_KIND
#define MMA_KIND 1   // FP8 = catalog cliff regime
#endif
#ifndef UNROLL
#define UNROLL 0     // 0 = #pragma unroll 1 (default)
#endif

#if MMA_KIND == 0
  #define MMA_M 64
  #define MMA_N 8
  #define TMEM_COLS 128
#elif MMA_KIND == 1
  #define MMA_M 128
  #define MMA_N 128
  #define TMEM_COLS 512
#elif MMA_KIND == 2
  #define MMA_M 128
  #define MMA_N 256
  #define TMEM_COLS 512
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int iters, int u1, int u2) {
    __shared__ __align__(1024) unsigned smem_A[3072];
    __shared__ __align__(1024) unsigned smem_B[3072];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < 3072; idx += 32) {
            smem_A[idx] = 0xDEADBEEFu ^ (idx * 0xCAFEBABEu);
            smem_B[idx] = 0x12345678u ^ (idx * 0x13579BDFu);
        }
    }
    if (threadIdx.x == 0) {
        tmem_slot = 0xFFFFFFFFu;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)));
    }
    __syncthreads();

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)), "n"(TMEM_COLS) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    __syncthreads();
    unsigned tmem_addr = tmem_slot;

#if MMA_KIND == 0
    // FP16: kind::f16, K=16
    unsigned idesc = (1U << 4)
                   | (1U << 7)
                   | (1U << 10)
                   | (((unsigned)MMA_N >> 3) << 17)
                   | (((unsigned)MMA_M >> 4) << 24);
#elif MMA_KIND == 1
    // FP8: kind::f8f6f4 with format e4m3 (binary 0,0,0 in low bits per mma_kind)
    // For FP8 e4m3 K=32 implicit
    unsigned idesc = (4U << 7)   // a_format = e4m3
                   | (4U << 10)  // b_format = e4m3
                   | (((unsigned)MMA_N >> 3) << 17)
                   | (((unsigned)MMA_M >> 4) << 24);
#elif MMA_KIND == 2
    // NVFP4 mxf4nvf4 block_scale.block16, K=64 implicit
    unsigned idesc = (5U << 7) | (5U << 10)
                   | (((unsigned)MMA_N >> 3) << 17)
                   | (((unsigned)MMA_M >> 4) << 24);
#endif

    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16;
    unsigned long long SBO = 2 * (unsigned long long)MMA_M;
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);

#if MMA_KIND == 2
    // NVFP4 needs scale factors in TMEM
    unsigned tsfa_addr = tmem_addr + 128;
    unsigned tsfb_addr = tmem_addr + 256;
    if (threadIdx.x == 0) {
        unsigned one_pack = 0x38383838u;
        for (int chunk = 0; chunk < 4; chunk++) {
            unsigned addr = tmem_addr + chunk * 128;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};\n"
                :: "r"(addr), "r"(one_pack), "r"(one_pack), "r"(one_pack), "r"(one_pack));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;");
    }
    __syncthreads();
#endif

    unsigned long long t0 = 0, t1 = 0;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (threadIdx.x == 0) {
        unsigned scaleC = 0;

#if UNROLL == 0
        #pragma unroll 1
#elif UNROLL == -1
        // no pragma -> compiler default
#else
        #pragma unroll UNROLL
#endif
        for (int i = 0; i < iters; i++) {
#if MMA_KIND == 0
            {
                unsigned z0 = 0, z1 = 0, z2 = 0, z3 = 0;
                asm volatile(
                    "{\n\t .reg .pred P;\n\t setp.ne.b32 P, %8, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%4, %5, %6, %7}, P;\n\t}"
                    :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                       "r"(z0), "r"(z1), "r"(z2), "r"(z3), "r"(scaleC)
                    : "memory");
            }
#elif MMA_KIND == 1
            asm volatile(
                "{\n\t .reg .pred P;\n\t setp.ne.b32 P, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, P;\n\t}"
                :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(scaleC)
                : "memory");
#elif MMA_KIND == 2
            asm volatile(
                "{\n\t .reg .pred P;\n\t setp.ne.b32 P, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
                "[%0], %1, %2, %3, [%5], [%6], P;\n\t}"
                :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                   "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr) : "memory");
#endif
            scaleC = 1;
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

    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr), "n"(TMEM_COLS));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long total_cy = t1 - t0;
        double cy_per_mma = (double)total_cy / (double)iters;
        ((unsigned long long*)C)[0] = total_cy;
        ((unsigned*)C)[2] = iters;
        printf("[KIND%d M=%d N=%d UNROLL=%d] iters=%d total_cy=%llu cy/MMA=%.3f\n",
               MMA_KIND, MMA_M, MMA_N, UNROLL, iters, total_cy, cy_per_mma);
    }
}
