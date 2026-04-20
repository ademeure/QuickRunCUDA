// Per-bit-position NVFP4 power: vary only one bit position at a time
// FP4 e2m1: bit 3=sign, bits 2-1=exp, bit 0=mantissa
// Compile: -H "#define K_SIZE 0/1" for K=64/96
// Args: -1 mode (0=all const, 1=bit0 random, 2=bit1 random, 4=bit2 random,
//                8=bit3 random, 15=all 4 random, 9=bit0+bit3 random,
//                10=bits 1,3, 12=bits 2,3, 7=bits 0,1,2 (no sign), 14=bits 1,2,3 (no mant))

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

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int iters, int bit_mask, int u2) {
    __shared__ __align__(1024) unsigned smem_A[3072];
    __shared__ __align__(1024) unsigned smem_B[3072];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    int smem_size = MMA_K * 16;

    // A always random
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            smem_A[idx] = 0xDEADBEEFu ^ idx * 0xCAFEBABEu;
        }
    }
    // B: each FP4 value has bits selectively varied
    // bit_mask: which bits vary (1=mant, 2=exp_low, 4=exp_high, 8=sign)
    // Other bits = 0x2 (= +1.0 in FP4 e2m1: sign=0, exp=01, mant=0)
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            unsigned r = 0xDEADBEEFu ^ idx * 0x13579BDFu;
            r ^= r >> 16; r *= 0xCAFEBABEu;
            // Build 32 bits: 8 FP4 values × 4 bits each
            unsigned val = 0;
            for (int p = 0; p < 8; p++) {
                // Default value for this FP4 = 0x2 (= +1.0)
                unsigned fp4 = 0x2;
                // Apply random bits as masked
                unsigned rb = (r >> (p * 4)) & 0xF;
                // For each bit in bit_mask, replace that bit with random
                if (bit_mask & 1) fp4 = (fp4 & ~1) | (rb & 1);
                if (bit_mask & 2) fp4 = (fp4 & ~2) | (rb & 2);
                if (bit_mask & 4) fp4 = (fp4 & ~4) | (rb & 4);
                if (bit_mask & 8) fp4 = (fp4 & ~8) | (rb & 8);
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
                "{\n\t .reg .pred P;\n\t setp.ne.b32 P, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], P;\n\t}"
                :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                   "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr) : "memory");
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
        printf("NVFP4 K=%d bit_mask=0x%X cy/MMA=%.2f\n",
               MMA_K, bit_mask, (double)(t1-t0)/iters);
    }
}
