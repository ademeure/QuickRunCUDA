// BF16 per-bit decomposition: force ONE bit position constant, others random.
// BF16 layout: bit 15=sign, bits 14:7=exp[7:0], bits 6:0=mant[6:0].
// Mode = 0..15 selects which B bit to force (constant 0).
// Mode = 100 + i: force B bit i constant 1.
// Mode = 200: B random (baseline).
// Mode = 300: B all-zero.
// Mode = 400 + i: force A bit i to 0 (B always random)
// Mode = 500 + i: force A bit i to 1 (B always random)
// Mode = 600..615: force B bits 0..i (cumulative low-to-high, 0=just bit 0, 15=all bits)
// Mode = 700..715: force B bits 15..(15-i) (cumulative high-to-low, 0=just sign, 15=all)
// Mode = 800: force B mantissa only (bits 0-6)
// Mode = 801: force B exp only (bits 7-14)
// Mode = 802: force B mant+exp (bits 0-14)
// Mode = 803: force B sign only (= mode 15)
// Mode = 804: force B sign+exp (bits 7-15)
// Mode = 805: force B sign+mant (bits 0-6,15)
//
// VERIFICATION: at startup, thread 0 of block 0 prints first 4 BF16 values
// of B (hex + decoded sign/exp/mant) so we can sanity-check encoding.

#define MMA_M 128
#define MMA_N 128
#define MMA_K 16
#ifndef MODE
#define MODE 200
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int iters, int mode, int verify) {
    __shared__ __align__(1024) unsigned smem_A[2048];
    __shared__ __align__(1024) unsigned smem_B[2048];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    // A pattern: random bytes, then force one bit position if mode 400+
    if (threadIdx.x < 32) {
        for (int i = 0; i < 64; i++) {
            unsigned idx = threadIdx.x + i*32;
            unsigned r_a = 0xDEADBEEFu ^ idx * 0xCAFEBABEu;
            unsigned w_a;
            if (mode >= 400 && mode <= 415) {
                int b = mode - 400;
                unsigned bm = (1u << b) | (1u << (b + 16));
                w_a = r_a & ~bm;  // force A bit b to 0
            } else if (mode >= 500 && mode <= 515) {
                int b = mode - 500;
                unsigned bm = (1u << b) | (1u << (b + 16));
                w_a = (r_a & ~bm) | bm;  // force A bit b to 1
            } else {
                w_a = r_a;
            }
            smem_A[idx] = w_a;
        }
    }

    // B pattern: random bytes, then force one bit position
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < 1024; idx += 32) {
            unsigned r = 0xDEADBEEFu ^ idx * 0x13579BDFu;
            unsigned w;
            if (mode == 200) {
                w = r;  // pure random
            } else if (mode == 300) {
                w = 0;  // all zero
            } else if (mode >= 0 && mode <= 15) {
                unsigned bit_mask = (1u << mode) | (1u << (mode + 16));
                w = r & ~bit_mask;
            } else if (mode >= 100 && mode <= 115) {
                int b = mode - 100;
                unsigned bit_mask = (1u << b) | (1u << (b + 16));
                w = (r & ~bit_mask) | bit_mask;
            } else if (mode >= 600 && mode <= 615) {
                // Cumulative low-to-high: force bits 0..(mode-600) to 0
                int top_bit = mode - 600;
                unsigned single_half = (1u << (top_bit + 1)) - 1;  // bits 0..top_bit set
                unsigned bit_mask = single_half | (single_half << 16);
                w = r & ~bit_mask;
            } else if (mode >= 700 && mode <= 715) {
                // Cumulative high-to-low: force bits (15-(mode-700))..15 to 0
                int n_bits = (mode - 700) + 1;
                unsigned single_half = ~((1u << (16 - n_bits)) - 1) & 0xFFFFu;  // top n_bits set
                unsigned bit_mask = single_half | (single_half << 16);
                w = r & ~bit_mask;
            } else if (mode == 800) {
                w = r & ~0x007F007Fu;  // mant only (bits 0-6)
            } else if (mode == 801) {
                w = r & ~0x7F807F80u;  // exp only (bits 7-14)
            } else if (mode == 802) {
                w = r & ~0x7FFF7FFFu;  // mant+exp (bits 0-14)
            } else if (mode == 803) {
                w = r & ~0x80008000u;  // sign only (bit 15)
            } else if (mode == 804) {
                w = r & ~0xFF80FF80u;  // sign+exp (bits 7-15)
            } else if (mode == 805) {
                w = r & ~0x807F807Fu;  // sign+mant (bits 0-6, 15)
            } else if (mode >= 900 && mode <= 1155) {
                // Force entire exp field (bits 7-14) to specific value V = mode - 900
                int V = mode - 900;
                if (V > 255) V = 255;
                unsigned exp_pat = (V & 0xFF) << 7;
                unsigned exp_mask = 0x7F80u;
                // Both halves of word
                w = (r & ~(exp_mask | (exp_mask << 16))) | (exp_pat | (exp_pat << 16));
            } else {
                // For mode >= 400 (A bit forcing), B is random
                w = r;
            }
            smem_B[idx] = w;
        }
    }
    if (threadIdx.x == 0) {
        tmem_slot = 0xFFFFFFFFu;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)));
    }
    __syncthreads();

    // VERIFY: print first 4 BF16 values of B if verify flag set
    if (verify && threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < 2; i++) {
            unsigned w = smem_B[i];
            unsigned bf16_lo = w & 0xFFFF;
            unsigned bf16_hi = (w >> 16) & 0xFFFF;
            printf("  B[idx=%d] word=0x%08x lo=0x%04x (s=%d e=%d m=0x%02x) hi=0x%04x (s=%d e=%d m=0x%02x)\n",
                   i, w, bf16_lo,
                   (bf16_lo >> 15) & 1, (bf16_lo >> 7) & 0xFF, bf16_lo & 0x7F,
                   bf16_hi,
                   (bf16_hi >> 15) & 1, (bf16_hi >> 7) & 0xFF, bf16_hi & 0x7F);
        }
    }

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
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16, SBO = 256;
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned disable_lane[4] = {0,0,0,0};

    unsigned long long t0=0, t1=0;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (threadIdx.x == 0) {
        unsigned enable_d = 0;
        for (int i = 0; i < iters; i++) {
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
        printf("BF16 perbit mode=%d iters=%d cy/MMA=%.2f\n",
               mode, iters, (double)(t1-t0)/iters);
    }
}
