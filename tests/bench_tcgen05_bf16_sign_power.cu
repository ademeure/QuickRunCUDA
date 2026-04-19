// BF16 sign-bit-specific power patterns for B operand.
// BF16 layout: bit 15 = sign, [14:7] = exp, [6:0] = mantissa.
// In packed unsigned (2 BF16 per word, little-endian):
//   word bits [15:0] = first BF16, [31:16] = second BF16.
//   sign mask = 0x80008000 (bit 15 of each BF16).
//
// SIGN_MODE selects what we do with B's sign bits (other 15 bits always random):
//   0 = sign random (BASELINE: B fully random)
//   1 = all-positive (sign = 0)
//   2 = all-negative (sign = 1)
//   3 = K-uniform-per-N (sign[k][n] = sign[0][n] for all k)
//   4 = N-uniform-per-K (sign[k][n] = sign[k][0] for all n)
//   5 = N-stride-32 sign (signs cycle every 32 N's: sign[k][n] = sign[k][n%32])
//   6 = N-stride-16 sign
//   7 = N-stride-8 sign
//   8 = N-stride-4 sign
//   9 = N-stride-2 sign
//   10 = K-uniform AND N-stride-32 (combined)

#define MMA_M 128
#define MMA_N 128
#define MMA_K 16
#define SIGN_MASK 0x80008000u   // bit 15 of each BF16 (packed 2 per word)

#ifndef SIGN_MODE
#define SIGN_MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int iters, int sign_mode, int u2) {
    __shared__ __align__(1024) unsigned smem_A[2048];
    __shared__ __align__(1024) unsigned smem_B[2048];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    // A always random
    if (threadIdx.x < 32) {
        for (int i = 0; i < 64; i++) {
            unsigned idx = threadIdx.x + i*32;
            smem_A[idx] = 0xDEADBEEFu ^ idx * 0xCAFEBABEu;
        }
    }

    // B layout: K=16 × N=128 BF16 → 1024 unsigned (each = 2 packed BF16)
    // smem_B[k * 64 + (n>>1)] holds n_even and n_odd BF16
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < 1024; idx += 32) {
            int k = idx / 64;
            int npair = idx % 64;       // 0..63 (each = 2 N values)
            int n_even = npair * 2;     // first BF16 in word

            // Random base for non-sign bits
            unsigned r = 0xDEADBEEFu ^ idx * 0x13579BDFu;
            // Apply sign mode
            unsigned sign_bits;   // value in SIGN_MASK positions
            switch (sign_mode) {
                case 0:
                    sign_bits = r & SIGN_MASK;          // random sign (from r)
                    break;
                case 1:
                    sign_bits = 0;                       // all positive
                    break;
                case 2:
                    sign_bits = SIGN_MASK;               // all negative
                    break;
                case 3: {
                    // K-uniform per N: sign[k][n] = sign[0][n] for all k
                    // Use only n in source idx (k=0)
                    unsigned r0 = 0xDEADBEEFu ^ (npair) * 0x13579BDFu;
                    sign_bits = r0 & SIGN_MASK;
                    break;
                }
                case 4: {
                    // N-uniform per K: sign[k][n] = sign[k][0] for all n
                    // sign for first BF16 in word follows k only; same for second BF16
                    unsigned r0 = 0xDEADBEEFu ^ (k * 64) * 0x13579BDFu;
                    unsigned s_low = r0 & 0x8000u;     // sign of n=0 for this k
                    sign_bits = s_low | (s_low << 16);  // both BF16 in word use same sign
                    break;
                }
                case 5: {
                    // N-stride-32 sign: sign[k][n] = sign[k][n%32]
                    // For this word at npair → n_even, n_odd. Source N = n_even % 32, n_odd % 32.
                    int ne_src = n_even % 32;
                    int no_src = (n_even + 1) % 32;
                    unsigned re = 0xDEADBEEFu ^ (k * 64 + (ne_src >> 1)) * 0x13579BDFu;
                    unsigned ro = 0xDEADBEEFu ^ (k * 64 + (no_src >> 1)) * 0x13579BDFu;
                    // Pick correct half from each
                    unsigned se = (ne_src & 1) ? ((re >> 16) & 0x8000u) : (re & 0x8000u);
                    unsigned so = (no_src & 1) ? ((ro >> 16) & 0x8000u) : (ro & 0x8000u);
                    sign_bits = se | (so << 16);
                    break;
                }
                case 6: {
                    int ne_src = n_even % 16;
                    int no_src = (n_even + 1) % 16;
                    unsigned re = 0xDEADBEEFu ^ (k * 64 + (ne_src >> 1)) * 0x13579BDFu;
                    unsigned ro = 0xDEADBEEFu ^ (k * 64 + (no_src >> 1)) * 0x13579BDFu;
                    unsigned se = (ne_src & 1) ? ((re >> 16) & 0x8000u) : (re & 0x8000u);
                    unsigned so = (no_src & 1) ? ((ro >> 16) & 0x8000u) : (ro & 0x8000u);
                    sign_bits = se | (so << 16);
                    break;
                }
                case 7: {
                    int ne_src = n_even % 8;
                    int no_src = (n_even + 1) % 8;
                    unsigned re = 0xDEADBEEFu ^ (k * 64 + (ne_src >> 1)) * 0x13579BDFu;
                    unsigned ro = 0xDEADBEEFu ^ (k * 64 + (no_src >> 1)) * 0x13579BDFu;
                    unsigned se = (ne_src & 1) ? ((re >> 16) & 0x8000u) : (re & 0x8000u);
                    unsigned so = (no_src & 1) ? ((ro >> 16) & 0x8000u) : (ro & 0x8000u);
                    sign_bits = se | (so << 16);
                    break;
                }
                case 8: {
                    int ne_src = n_even % 4;
                    int no_src = (n_even + 1) % 4;
                    unsigned re = 0xDEADBEEFu ^ (k * 64 + (ne_src >> 1)) * 0x13579BDFu;
                    unsigned ro = 0xDEADBEEFu ^ (k * 64 + (no_src >> 1)) * 0x13579BDFu;
                    unsigned se = (ne_src & 1) ? ((re >> 16) & 0x8000u) : (re & 0x8000u);
                    unsigned so = (no_src & 1) ? ((ro >> 16) & 0x8000u) : (ro & 0x8000u);
                    sign_bits = se | (so << 16);
                    break;
                }
                case 9: {
                    // Stride 2: pairs of N share sign
                    // ne and no in same pair → both use sign of n_even
                    int n_grp = n_even & ~1;
                    unsigned r0 = 0xDEADBEEFu ^ (k * 64 + (n_grp >> 1)) * 0x13579BDFu;
                    unsigned s = r0 & 0x8000u;  // sign of n_even
                    sign_bits = s | (s << 16);
                    break;
                }
                case 10: {
                    // K-uniform + N-stride-32 sign
                    int n_src_e = n_even % 32;
                    int n_src_o = (n_even + 1) % 32;
                    unsigned re = 0xDEADBEEFu ^ (n_src_e >> 1) * 0x13579BDFu;
                    unsigned ro = 0xDEADBEEFu ^ (n_src_o >> 1) * 0x13579BDFu;
                    unsigned se = (n_src_e & 1) ? ((re >> 16) & 0x8000u) : (re & 0x8000u);
                    unsigned so = (n_src_o & 1) ? ((ro >> 16) & 0x8000u) : (ro & 0x8000u);
                    sign_bits = se | (so << 16);
                    break;
                }
                case 11: {
                    // N-stride 64 sign
                    int ne_src = n_even % 64;
                    int no_src = (n_even + 1) % 64;
                    unsigned re = 0xDEADBEEFu ^ (k * 64 + (ne_src >> 1)) * 0x13579BDFu;
                    unsigned ro = 0xDEADBEEFu ^ (k * 64 + (no_src >> 1)) * 0x13579BDFu;
                    unsigned se = (ne_src & 1) ? ((re >> 16) & 0x8000u) : (re & 0x8000u);
                    unsigned so = (no_src & 1) ? ((ro >> 16) & 0x8000u) : (ro & 0x8000u);
                    sign_bits = se | (so << 16);
                    break;
                }
                case 12: {
                    // K-uniform + N-stride-16
                    int n_src_e = n_even % 16;
                    int n_src_o = (n_even + 1) % 16;
                    unsigned re = 0xDEADBEEFu ^ (n_src_e >> 1) * 0x13579BDFu;
                    unsigned ro = 0xDEADBEEFu ^ (n_src_o >> 1) * 0x13579BDFu;
                    unsigned se = (n_src_e & 1) ? ((re >> 16) & 0x8000u) : (re & 0x8000u);
                    unsigned so = (n_src_o & 1) ? ((ro >> 16) & 0x8000u) : (ro & 0x8000u);
                    sign_bits = se | (so << 16);
                    break;
                }
                case 13: {
                    // K-uniform + N-stride-64
                    int n_src_e = n_even % 64;
                    int n_src_o = (n_even + 1) % 64;
                    unsigned re = 0xDEADBEEFu ^ (n_src_e >> 1) * 0x13579BDFu;
                    unsigned ro = 0xDEADBEEFu ^ (n_src_o >> 1) * 0x13579BDFu;
                    unsigned se = (n_src_e & 1) ? ((re >> 16) & 0x8000u) : (re & 0x8000u);
                    unsigned so = (n_src_o & 1) ? ((ro >> 16) & 0x8000u) : (ro & 0x8000u);
                    sign_bits = se | (so << 16);
                    break;
                }
                // Block-uniform sign: within each block of BSZ N positions,
                // all share ONE single sign value. Different blocks have
                // independent (random) sign. # unique signs per K row = N/BSZ.
                case 20: {  // BSZ=2 → 64 unique signs
                    int blk_e = n_even / 2;
                    int blk_o = (n_even + 1) / 2;
                    unsigned re = 0xDEADBEEFu ^ (k * 64 + blk_e) * 0x13579BDFu;
                    unsigned ro = 0xDEADBEEFu ^ (k * 64 + blk_o) * 0x13579BDFu;
                    sign_bits = (re & 0x8000u) | ((ro & 0x8000u) << 16);
                    break;
                }
                case 21: {  // BSZ=4 → 32 unique
                    int blk_e = n_even / 4;
                    int blk_o = (n_even + 1) / 4;
                    unsigned re = 0xDEADBEEFu ^ (k * 64 + blk_e) * 0x13579BDFu;
                    unsigned ro = 0xDEADBEEFu ^ (k * 64 + blk_o) * 0x13579BDFu;
                    sign_bits = (re & 0x8000u) | ((ro & 0x8000u) << 16);
                    break;
                }
                case 22: {  // BSZ=8 → 16 unique
                    int blk_e = n_even / 8;
                    int blk_o = (n_even + 1) / 8;
                    unsigned re = 0xDEADBEEFu ^ (k * 64 + blk_e) * 0x13579BDFu;
                    unsigned ro = 0xDEADBEEFu ^ (k * 64 + blk_o) * 0x13579BDFu;
                    sign_bits = (re & 0x8000u) | ((ro & 0x8000u) << 16);
                    break;
                }
                case 23: {  // BSZ=16 → 8 unique
                    int blk_e = n_even / 16;
                    int blk_o = (n_even + 1) / 16;
                    unsigned re = 0xDEADBEEFu ^ (k * 64 + blk_e) * 0x13579BDFu;
                    unsigned ro = 0xDEADBEEFu ^ (k * 64 + blk_o) * 0x13579BDFu;
                    sign_bits = (re & 0x8000u) | ((ro & 0x8000u) << 16);
                    break;
                }
                case 24: {  // BSZ=32 → 4 unique
                    int blk_e = n_even / 32;
                    int blk_o = (n_even + 1) / 32;
                    unsigned re = 0xDEADBEEFu ^ (k * 64 + blk_e) * 0x13579BDFu;
                    unsigned ro = 0xDEADBEEFu ^ (k * 64 + blk_o) * 0x13579BDFu;
                    sign_bits = (re & 0x8000u) | ((ro & 0x8000u) << 16);
                    break;
                }
                case 25: {  // BSZ=64 → 2 unique
                    int blk_e = n_even / 64;
                    int blk_o = (n_even + 1) / 64;
                    unsigned re = 0xDEADBEEFu ^ (k * 64 + blk_e) * 0x13579BDFu;
                    unsigned ro = 0xDEADBEEFu ^ (k * 64 + blk_o) * 0x13579BDFu;
                    sign_bits = (re & 0x8000u) | ((ro & 0x8000u) << 16);
                    break;
                }
                default: sign_bits = r & SIGN_MASK; break;
            }
            // Combine: non-sign bits from r, sign from sign_bits
            smem_B[idx] = (r & ~SIGN_MASK) | sign_bits;
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

    // BF16: a_format=1, b_format=1
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
        printf("BF16 sign sign_mode=%d iters=%d cy/MMA=%.2f\n",
               sign_mode, iters, (double)(t1-t0)/iters);
    }
}
