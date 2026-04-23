// Correctness check: verify K=96 ULTRA actually consumes K=96 data, not K=64.
//
// Strategy:
//   - Set up smem_A and smem_B large enough for K=96.
//   - For BOTH K=64 and K=96: pre-fill the K∈[64,96) region of B with a
//     non-zero, distinctive pattern; pre-fill K∈[0,64) with zeros.
//   - K=64 inst will read only K∈[0,64) → all zeros → result tile = 0.
//   - K=96 inst will additionally read K∈[64,96) (non-zero) → result ≠ 0.
//   - Single MMA per kernel (iters=1), accumulator zeroed.
//   - Read out a few TMEM accumulator entries and report.
//
// Args:
//   -0  iters  (kept at 1)
//   -1  mode   64 -> K=64, 96 -> K=96
//   -2  unused

#define MMA_M 256
#define MMA_N 256

extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void kernel(float* A, float* B, float* C, int iters, int mode, int u2) {
    __shared__ __align__(1024) unsigned smem_A[3072];
    __shared__ __align__(1024) unsigned smem_B[3072];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    // For K=64 layout: 64*256/8 = 2048 dwords. K=96: 96*256/8 = 3072 dwords.
    // We init A as: K∈[0,64) all 0x22 (FP4 +1.0), K∈[64,96) all 0x44 (FP4 +2.0).
    // We init B as: K∈[0,64) all 0x00 (FP4 +0.0), K∈[64,96) all 0x22 (FP4 +1.0).
    // K=64 reads K∈[0,64): A*B = sum(1.0 * 0.0) = 0.0
    // K=96 reads K∈[0,96): K<64 → 1.0*0.0 = 0; K∈[64,96) → 2.0*1.0 = 2.0
    //    Per (m,n): 32 K-positions × 2.0 = 64.0 contribution
    // Each block_scale block16 SF byte = 0x38 (UE4M3 = 1.0)

    // Determine layout: K=96 layout is the union; we put K=0..63 first, K=64..95 second.
    // Layout: A is indexed [m*K/8 + k_pack], B is [k*N/8 + n_pack] in row-K-major.
    // smem_A[idx]:  m = idx / (K/8); k_pack = idx % (K/8)
    // smem_B[idx]:  k = idx / (N/8); n_pack = idx % (N/8)

    int npacks = MMA_N / 8;     // 32
    int kpacks = 96 / 8;        // 12  (0..7 = K[0..63] 8 dwords, 8..11 = K[64..95] 4 dwords)
    // For A: m=0..255, k_pack=0..(MMA_K/8)-1 where MMA_K=64 or 96
    // Test: distinguish K=64 vs K=96 hardware behavior
    // A = all 0x22 (FP4 +1.0) everywhere
    // B = K[0,64): 0x22 (+1.0), K[64,96): 0x44 (+2.0)
    // K=64 inst expected result per (m,n): 64 × 1.0 × 1.0 = 64.0
    // K=96 inst expected result per (m,n): 64 × 1.0 + 32 × 1.0 × 2.0 = 64 + 64 = 128.0
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < 3072; idx += 32) {
            smem_A[idx] = 0x22222222u;  // A: FP4 +1.0 everywhere
            // B layout: idx = k * (N/8) + n_pack ; N/8 = 32 ; so k = idx / 32
            int k = idx / 32;
            unsigned val_b;
            if (k < 64) val_b = 0x22222222u;  // FP4 +1.0
            else        val_b = 0x44444444u;  // FP4 +2.0
            smem_B[idx] = val_b;
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

    // SF region: all 0x38 (UE4M3 1.0)
    {
        unsigned one = 0x38383838u;
        for (int chunk = 0; chunk < 4; chunk++) {
            unsigned addr = tmem_addr + chunk * 128;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};\n"
                :: "r"(addr), "r"(one), "r"(one), "r"(one), "r"(one));
        }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;");
    __syncthreads();

    unsigned idesc_base = (5U << 7) | (5U << 10)
                        | (((unsigned)MMA_N >> 3) << 17)
                        | (((unsigned)MMA_M >> 4) << 24);
    unsigned idesc = (mode == 96) ? (idesc_base | (1U << 31)) : idesc_base;

    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16;
    unsigned long long SBO = 2 * (unsigned long long)(MMA_M / 2);
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);

    if ((blockIdx.x % 2) == 0 && threadIdx.x == 0) {
        unsigned scaleC = 0;  // accumulator zeroed
        for (int i = 0; i < iters; i++) {
            asm volatile(
                "{\n\t .reg .pred PRED;\n\t setp.ne.b32 PRED, %4, 0;\n\t"
                "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], PRED;\n\t}"
                :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                   "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr) : "memory");
            scaleC = 1;
        }
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
        unsigned phase_w = 0;
        asm volatile("{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t @P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase_w));
    }
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");

    // Read out part of the TMEM accumulator
    // tcgen05.ld returns 128 columns × 32-bit per row of TMEM tile.
    // Use 32x32b.x4: 4 dwords per thread = 128 b returned per call.
    if ((blockIdx.x % 2) == 0) {
        unsigned r0, r1, r2, r3;
        unsigned read_addr = tmem_addr + 0;  // first column block
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(read_addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            float f0 = __int_as_float(r0);
            float f1 = __int_as_float(r1);
            float f2 = __int_as_float(r2);
            float f3 = __int_as_float(r3);
            printf("MODE=%d acc[0..3]=%.4f %.4f %.4f %.4f  (raw 0x%08x 0x%08x 0x%08x 0x%08x)\n",
                mode, f0, f1, f2, f3, r0, r1, r2, r3);
        }
        // Also store to global C
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            ((unsigned*)C)[0] = r0;
            ((unsigned*)C)[1] = r1;
            ((unsigned*)C)[2] = r2;
            ((unsigned*)C)[3] = r3;
        }
    }

    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
}
