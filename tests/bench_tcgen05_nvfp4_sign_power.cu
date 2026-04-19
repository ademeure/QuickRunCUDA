// NVFP4 (kind::mxf4nvf4.block_scale.block16) SIGN-BIT-ONLY power microbench
// FP4 e2m1: bit 3=sign, bits 2-1=exp, bit 0=mantissa.
// Each unsigned word = 8 FP4 values; sign mask = 0x88888888.
// SF=1.0 always. A always random. B sign bits varied; other 3 bits always random.
//
// SIGN_MODE for B (analogous to BF16 sign kernel):
//   0 = sign random (BASELINE)
//   1 = all-positive (sign=0)
//   2 = all-negative (sign=1)
//   30 = TRUE period-2 ABABAB... (random A, B per K)
//   31 = FORCED +-+-+- (sign=0,1,0,1,...)
//   4 = single sign per K (one random sign across all 128 N for each k)
//   3 = K-uniform per N (sign[k][n] = sign[0][n] for all k)
//   5..11 = period-X (X=32, 16, 8, 4, 2, 64, 128)
//   20..25 = block-uniform BSZ=2,4,8,16,32,64

#define MMA_M 128
#define MMA_N 128
#define MMA_K 64
#define SIGN_MASK 0x88888888u   // bit 3 of each FP4 (8 FP4 per word)
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
    // B layout: K=64 × N=128 FP4 = 4KB. Each unsigned = 8 FP4 (8 N positions).
    // smem_B[k * 16 + n/8] holds N[n..n+7], 8 FP4 values packed.
    // Total = 64 * 16 = 1024 unsigned.
    // Sign of FP4 at position n (within word): bit (4*(n%8) + 3)
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < 1024; idx += 32) {
            int k = idx / 16;
            int npack = idx % 16;     // 0..15 (each pack = 8 N values)
            int n_base = npack * 8;   // first N in this pack: 0,8,...120
            unsigned r = 0xDEADBEEFu ^ idx * 0x13579BDFu;  // random base for non-sign

            // Build 8-bit sign mask for this word (one bit per FP4 position)
            // Then expand to 0x88888888-style placement (bit 3, 7, 11, 15, 19, 23, 27, 31)
            unsigned sign_set = 0;  // 8 bits, one per FP4 in this word

            switch (sign_mode) {
                case 0:
                    sign_set = (r >> 3) & 0xFF;   // random 8 sign bits
                    break;
                case 1: sign_set = 0x00; break;
                case 2: sign_set = 0xFF; break;
                case 30: {
                    // TRUE period-2 ABABAB random A,B per K
                    unsigned r_a = 0xDEADBEEFu ^ (k * 2 + 0) * 0x13579BDFu;
                    unsigned r_b = 0xDEADBEEFu ^ (k * 2 + 1) * 0x13579BDFu;
                    unsigned sa = (r_a >> 15) & 1;
                    unsigned sb = (r_b >> 15) & 1;
                    // Pack: even N positions=A, odd=B
                    sign_set = (sa<<0) | (sb<<1) | (sa<<2) | (sb<<3) | (sa<<4) | (sb<<5) | (sa<<6) | (sb<<7);
                    break;
                }
                case 31:
                    // FORCED +-+-+- (sign at even N=0, odd N=1)
                    sign_set = 0xAA;  // 1010_1010 = bit at odd positions set
                    break;
                case 4: {
                    // Single sign per K row
                    unsigned r0 = 0xDEADBEEFu ^ (k) * 0x13579BDFu;
                    unsigned s = (r0 >> 15) & 1;
                    sign_set = s ? 0xFF : 0x00;
                    break;
                }
                case 3: {
                    // K-uniform per N: sign[k][n] = sign[0][n]
                    unsigned r0 = 0xDEADBEEFu ^ (npack) * 0x13579BDFu;
                    sign_set = (r0 >> 3) & 0xFF;
                    break;
                }
                // Period-X modes: sign[k][n] = sign[k][n%X]
                case 5: case 6: case 7: case 8: case 9: case 10: case 11: {
                    int X = (sign_mode == 5) ? 32 : (sign_mode == 6) ? 16 :
                             (sign_mode == 7) ? 8 : (sign_mode == 8) ? 4 :
                             (sign_mode == 9) ? 2 : (sign_mode == 10) ? 64 : 128;
                    for (int p = 0; p < 8; p++) {
                        int n = n_base + p;
                        int n_src = n % X;
                        unsigned rs = 0xDEADBEEFu ^ (k * (X) + n_src) * 0x13579BDFu;
                        unsigned s = (rs >> 15) & 1;
                        sign_set |= (s << p);
                    }
                    break;
                }
                // Block-uniform modes: each block of BSZ N's shares one random sign
                case 20: case 21: case 22: case 23: case 24: case 25: {
                    int BSZ = 1 << (sign_mode - 19);  // 2,4,8,16,32,64
                    for (int p = 0; p < 8; p++) {
                        int n = n_base + p;
                        int blk = n / BSZ;
                        unsigned rs = 0xDEADBEEFu ^ (k * 128 + blk) * 0x13579BDFu;
                        unsigned s = (rs >> 15) & 1;
                        sign_set |= (s << p);
                    }
                    break;
                }
                default: sign_set = (r >> 3) & 0xFF; break;
            }
            // Expand 8-bit sign_set to 32-bit position (bit p of sign_set → bit (4*p+3) of word)
            unsigned sign_bits = 0;
            for (int p = 0; p < 8; p++) {
                if ((sign_set >> p) & 1) sign_bits |= (1u << (4*p + 3));
            }
            // Combine: non-sign bits from r, sign from computed
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
    unsigned tsfa_addr = tmem_addr + 128;
    unsigned tsfb_addr = tmem_addr + 256;

    // Init TMEM with UE4M3 1.0 (byte 0x38)
    {
        unsigned one_pack = 0x38383838u;
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
                   | (((unsigned)MMA_M >> 4) << 24);

    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16, SBO = 256;
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);

    unsigned long long t0=0, t1=0;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (threadIdx.x == 0) {
        unsigned scaleC = 0;
        for (int i = 0; i < iters; i++) {
            asm volatile(
                "{\n\t .reg .pred PRED;\n\t setp.ne.b32 PRED, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], PRED;\n\t}"
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
            "{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        printf("NVFP4 sign sign_mode=%d iters=%d cy/MMA=%.2f\n",
               sign_mode, iters, (double)(t1-t0)/iters);
    }
}
