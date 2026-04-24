// D5 verification: tcgen05.mma cy/MMA across formats at M=128, N variable
// Mode (-1):
//   0 = FP16   (kind::f16,        K=16,  a/b_format=0)
//   1 = FP8    (kind::f8f6f4,     K=32,  E4M3 a/b_format=0)
//   2 = FP4    (kind::mxf4nvf4,   K=64,  block_scale.block16, E2M1 a/b_format=5)
// MMA_N injected via -H "#define MMA_N 256"
//
// Args:
//   -0 (iters)
//   -1 (mode 0/1/2)
//   -2 unused
//
// Single CTA, 32 threads. All MMAs issued by one warp (catalog convention).

#ifndef MMA_M
#define MMA_M 128
#endif
#ifndef MMA_N
#define MMA_N 256
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int iters, int mode, int u2) {
    __shared__ __align__(1024) unsigned smem_A[3072];
    __shared__ __align__(1024) unsigned smem_B[3072];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    int MMA_K;
    if (mode == 0)      MMA_K = 16;     // FP16
    else if (mode == 1) MMA_K = 32;     // FP8
    else                MMA_K = 64;     // FP4

    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < 3072; idx += 32) {
            unsigned r = (idx + 0xC0FFEE00u) * 0x9E3779B1u;
            r ^= r >> 16; r *= 0x85EBCA6Bu;
            r ^= r >> 13; r *= 0xC2B2AE35u;
            r ^= r >> 16;
            smem_A[idx] = r;
            smem_B[idx] = r ^ 0xDEADBEEFu;
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

    // FP4 needs SF region populated with UE4M3 1.0 (0x38)
    if (mode == 2) {
        unsigned one_pack = 0x38383838u;
        for (int chunk = 0; chunk < 4; chunk++) {
            unsigned col_base = chunk * 128 + (threadIdx.x * 4);
            unsigned addr = tmem_addr + col_base;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};\n"
                :: "r"(addr), "r"(one_pack), "r"(one_pack), "r"(one_pack), "r"(one_pack));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;");
    }
    __syncthreads();

    unsigned idesc;
    if (mode == 0) {
        // FP16 (kind::f16) — a/b_format=0, sparse=0
        idesc = (1U << 4) | (1U << 7) | (1U << 10)
              | (((unsigned)MMA_N >> 3) << 17)
              | (((unsigned)MMA_M >> 4) << 24);
    } else if (mode == 1) {
        // FP8 E4M3 (kind::f8f6f4) — a_format=0,b_format=0
        idesc = (((unsigned)MMA_N >> 3) << 17)
              | (((unsigned)MMA_M >> 4) << 24);
    } else {
        // FP4 E2M1 (kind::mxf4nvf4 block_scale.block16) — a/b_format=5, K_SIZE bit31=0 (K=64)
        idesc = (5U << 7) | (5U << 10)
              | (((unsigned)MMA_N >> 3) << 17)
              | (((unsigned)MMA_M >> 4) << 24);
    }

    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16;
    unsigned long long SBO = 256;
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned disable_lane[4] = {0,0,0,0};

    if (threadIdx.x == 0) {
        // Warmup
        unsigned enable_d = 0;
        for (int i = 0; i < 8; i++) {
            if (mode == 0) {
                asm volatile(
                    "{\n\t .reg .pred PRED;\n\t setp.ne.b32 PRED, %8, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%4, %5, %6, %7}, PRED;\n\t}"
                    :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                       "r"(disable_lane[0]), "r"(disable_lane[1]), "r"(disable_lane[2]), "r"(disable_lane[3]),
                       "r"(enable_d) : "memory");
            } else if (mode == 1) {
                asm volatile(
                    "{\n\t .reg .pred PRED;\n\t setp.ne.b32 PRED, %8, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%4, %5, %6, %7}, PRED;\n\t}"
                    :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                       "r"(disable_lane[0]), "r"(disable_lane[1]), "r"(disable_lane[2]), "r"(disable_lane[3]),
                       "r"(enable_d) : "memory");
            } else {
                asm volatile(
                    "{\n\t .reg .pred P;\n\t setp.ne.b32 P, %6, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], P;\n\t}"
                    :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                       "r"(tsfa_addr), "r"(tsfb_addr), "r"(enable_d) : "memory");
            }
            enable_d = 1;
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
        unsigned phase = 0;
        asm volatile(
            "{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase));

        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        for (int i = 0; i < iters; i++) {
            if (mode == 0) {
                asm volatile(
                    "{\n\t .reg .pred PRED;\n\t setp.ne.b32 PRED, 1, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%4, %5, %6, %7}, PRED;\n\t}"
                    :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                       "r"(disable_lane[0]), "r"(disable_lane[1]), "r"(disable_lane[2]), "r"(disable_lane[3])
                    : "memory");
            } else if (mode == 1) {
                asm volatile(
                    "{\n\t .reg .pred PRED;\n\t setp.ne.b32 PRED, 1, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%4, %5, %6, %7}, PRED;\n\t}"
                    :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                       "r"(disable_lane[0]), "r"(disable_lane[1]), "r"(disable_lane[2]), "r"(disable_lane[3])
                    : "memory");
            } else {
                asm volatile(
                    "{\n\t .reg .pred P;\n\t setp.ne.b32 P, 1, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], P;\n\t}"
                    :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                       "r"(tsfa_addr), "r"(tsfb_addr) : "memory");
            }
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
        phase = 1;
        asm volatile(
            "{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

        unsigned long long total = t1 - t0;
        printf("D5 mode=%d M=%d N=%d K=%d iters=%d total_cy=%llu cy/MMA=%.2f\n",
               mode, MMA_M, MMA_N, MMA_K, iters, total, (double)total / iters);
        ((unsigned long long*)C)[0] = total;
    }
    __syncthreads();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
}
