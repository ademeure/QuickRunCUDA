// NVFP4 K=96, single tcgen05 — RANDOM B with sparsity + optional sign-match.
// Args:
//   u0 = iters
//   u1 = sparsity_pct (0..100, % of B elements set to zero)
//   u2 = match_offset (0 = zero element sign = 0;
//                      64 = zero element sign = sign of element at N-64;
//                      negative or other = baseline random sign for zero too)
//
// A is fully random. B is fully random base, then sparsity_pct % of FP4 elements
// are forced to zero (magnitude=0). For those zero elements, the sign bit is set
// according to match_offset.
#define MMA_M 256
#define MMA_N 256
#define MMA_K 96

extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void kernel(float* A, float* B, float* C, int iters, int sparsity_pct, int match_offset) {
    __shared__ __align__(1024) unsigned smem_A[3072];
    __shared__ __align__(1024) unsigned smem_B[3072];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    int smem_size = MMA_K * MMA_N / 8;   // 3072 packed dwords (8 FP4 each)
    int npacks = MMA_N / 8;              // 32 packs per K-row

    // === A: fully random (k, n) pattern. Each pack holds 8 FP4 values.
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            unsigned r = (idx + blockIdx.x * 1024u) * 0x9E3779B1u;
            r ^= r >> 16; r *= 0x85EBCA6Bu;
            r ^= r >> 13; r *= 0xC2B2AE35u;
            r ^= r >> 16;
            smem_A[idx] = r;            // all 32 bits random → 8 random FP4
        }
    }

    // === B: random base, then per-FP4-element sparsity decision, sign-match for zero
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            int k = idx / npacks;
            int npack = idx % npacks;       // which 8-element column group
            int n_base = npack * 8;
            // Base random bits for this dword (8 FP4 = 8×4 bits)
            unsigned r = (idx + blockIdx.x * 1024u + 0xC0FFEE00u) * 0x9E3779B1u;
            r ^= r >> 16; r *= 0x85EBCA6Bu;
            r ^= r >> 13; r *= 0xC2B2AE35u;
            r ^= r >> 16;
            // PASS 1: compute base FP4 (random + sparsity-forced-zero with original sign)
            // Sparsity-forced elements get mag=0 but KEEP their random sign for now.
            unsigned base_fp4[8];
            for (int p = 0; p < 8; p++) {
                int n = n_base + p;
                unsigned fp4 = (r >> (p * 4)) & 0xF;
                unsigned spr_h = ((unsigned)k * 1664525u + (unsigned)n * 1013904223u + 0xFEEDFACEu);
                spr_h ^= spr_h >> 16; spr_h *= 0xCAFEBABEu;
                spr_h ^= spr_h >> 13;
                bool is_sparse = ((int)(spr_h % 100u)) < sparsity_pct;
                if (is_sparse) {
                    fp4 = fp4 & 0x8;   // keep sign bit, zero out magnitude
                }
                base_fp4[p] = fp4;
            }
            // PASS 2: for ALL on-bus zero-magnitude elements (mag low 3 bits == 0),
            // apply match_offset sign policy. Otherwise leave sign as base.
            unsigned val = 0;
            for (int p = 0; p < 8; p++) {
                int n = n_base + p;
                unsigned fp4 = base_fp4[p];
                unsigned mag = fp4 & 0x7;
                if (mag == 0) {
                    unsigned sign_bit = 0;
                    if (match_offset > 0) {
                        // Look at element at (k, n - match_offset). Wrap modulo MMA_N.
                        int n_neighbor = (n - match_offset + MMA_N) % MMA_N;
                        int npack_n = n_neighbor / 8;
                        int p_n = n_neighbor % 8;
                        int idx_n = k * npacks + npack_n;
                        unsigned r_n = (idx_n + blockIdx.x * 1024u + 0xC0FFEE00u) * 0x9E3779B1u;
                        r_n ^= r_n >> 16; r_n *= 0x85EBCA6Bu;
                        r_n ^= r_n >> 13; r_n *= 0xC2B2AE35u;
                        r_n ^= r_n >> 16;
                        unsigned fp4_n_base = (r_n >> (p_n * 4)) & 0xF;
                        // Apply same sparsity logic to neighbor: mag→0 if sparse-decided
                        unsigned spr_n = ((unsigned)k * 1664525u + (unsigned)n_neighbor * 1013904223u + 0xFEEDFACEu);
                        spr_n ^= spr_n >> 16; spr_n *= 0xCAFEBABEu;
                        spr_n ^= spr_n >> 13;
                        bool n_sparse = ((int)(spr_n % 100u)) < sparsity_pct;
                        unsigned fp4_n_eff = n_sparse ? (fp4_n_base & 0x8) : fp4_n_base;
                        // If neighbor's effective magnitude is 0, its sign is 0 (this n→too)
                        // (avoids one-level recursion ambiguity)
                        if ((fp4_n_eff & 0x7) == 0) {
                            sign_bit = 0;
                        } else {
                            sign_bit = (fp4_n_eff >> 3) & 1;
                        }
                    } else if (match_offset < 0) {
                        // Keep base random sign (no override)
                        sign_bit = (fp4 >> 3) & 1;
                    } else {
                        sign_bit = 0;       // mo=0: force ALL zero-mag signs to 0
                    }
                    fp4 = sign_bit << 3;
                }
                val |= (fp4 << (p * 4));
            }
            smem_B[idx] = val;
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

    {
        unsigned one_pack = 0x38383838u;   // FP4 SF = 1.0 (UE4M3 0x38)
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
                   | (1U << 31);

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
        unsigned scaleC = 0;
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
        asm volatile(
            "{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase_w));
    }
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
}
