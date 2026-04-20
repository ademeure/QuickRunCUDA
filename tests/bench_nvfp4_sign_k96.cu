// NVFP4 sign-bit power microbench supporting K=64 and K=96
// Compile-time: -H "#define K_SIZE 0" for K64, "#define K_SIZE 1" for K96

#ifndef MMA_M
#define MMA_M 128
#endif
#ifndef MMA_N
#define MMA_N 128
#endif
#ifndef K_SIZE
#define K_SIZE 0
#endif
#if K_SIZE == 0
#define MMA_K 64
#else
#define MMA_K 96
#endif

#define SIGN_MASK 0x88888888u

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int iters, int sign_mode, int u2) {
    __shared__ __align__(1024) unsigned smem_A[3072];   // big enough for K=96
    __shared__ __align__(1024) unsigned smem_B[3072];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    // SMEM size: K * N(128) FP4 / 8 FP4 per word = K * 16 unsigned
    int smem_size = MMA_K * 16;  // K=64 → 1024, K=96 → 1536

    // A: random
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            smem_A[idx] = 0xDEADBEEFu ^ idx * 0xCAFEBABEu;
        }
    }
    // B: build with controlled sign bits
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            int k = idx / 16;
            int npack = idx % 16;
            int n_base = npack * 8;
            unsigned r = 0xDEADBEEFu ^ idx * 0x13579BDFu;
            unsigned sign_set = 0;

            switch (sign_mode) {
                case 0: sign_set = (r >> 3) & 0xFF; break;          // random
                case 1: sign_set = 0x00; break;                       // all-positive
                case 2: sign_set = 0xFF; break;                       // all-negative
                case 3: { // K-uniform per N
                    unsigned r0 = 0xDEADBEEFu ^ (npack) * 0x13579BDFu;
                    sign_set = (r0 >> 3) & 0xFF;
                    break;
                }
                case 4: { // single sign per K row (constant per K)
                    unsigned r0 = 0xDEADBEEFu ^ (k) * 0x13579BDFu;
                    unsigned s = (r0 >> 15) & 1;
                    sign_set = s ? 0xFF : 0x00;
                    break;
                }
                case 30: { // True period-2 ABABAB random A,B per K row
                    unsigned r_a = 0xDEADBEEFu ^ (k * 2 + 0) * 0x13579BDFu;
                    unsigned r_b = 0xDEADBEEFu ^ (k * 2 + 1) * 0x13579BDFu;
                    unsigned sa = (r_a >> 15) & 1;
                    unsigned sb = (r_b >> 15) & 1;
                    sign_set = (sa<<0) | (sb<<1) | (sa<<2) | (sb<<3) | (sa<<4) | (sb<<5) | (sa<<6) | (sb<<7);
                    break;
                }
                case 31: sign_set = 0xAA; break;                      // forced +-+-+-
                // Period-X N-direction patterns
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
                // Block-uniform N: each block of BSZ N's shares one sign
                case 20: case 21: case 22: case 23: case 24: case 25: {
                    int BSZ = 1 << (sign_mode - 19);
                    for (int p = 0; p < 8; p++) {
                        int n = n_base + p;
                        int blk = n / BSZ;
                        unsigned rs = 0xDEADBEEFu ^ (k * 128 + blk) * 0x13579BDFu;
                        unsigned s = (rs >> 15) & 1;
                        sign_set |= (s << p);
                    }
                    break;
                }
                // K-direction patterns at fixed N
                case 40: { // sign[k][n] depends only on k%2 (period 2 along K)
                    unsigned rs = 0xDEADBEEFu ^ (k % 2) * 0x13579BDFu;
                    sign_set = (rs >> 3) & 0xFF;
                    break;
                }
                case 41: { // sign[k][n] depends on k%4 (period 4 along K)
                    unsigned rs = 0xDEADBEEFu ^ (k % 4) * 0x13579BDFu;
                    sign_set = (rs >> 3) & 0xFF;
                    break;
                }
                default: sign_set = (r >> 3) & 0xFF; break;
            }
            unsigned sign_bits = 0;
            for (int p = 0; p < 8; p++) {
                if ((sign_set >> p) & 1) sign_bits |= (1u << (4*p + 3));
            }
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
                   | (((unsigned)MMA_M >> 4) << 24)
                   | (((unsigned)K_SIZE) << 31);

    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
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
        printf("NVFP4 K=%d sign_mode=%d iters=%d cy/MMA=%.2f\n",
               MMA_K, sign_mode, iters, (double)(t1-t0)/iters);
    }
}
