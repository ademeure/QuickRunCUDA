// NVFP4 full power exploration: K={64,96}, SF mode, sign mode, MMA dims, 1/2 CTA
// Compile-time:
//   -H "#define K_SIZE 0/1" → K=64/96
//   -H "#define CTA_GROUP 1/2"  → single or 2-CTA cluster
//   -H "#define MMA_M ..."  → 64/128/256 (256 only with CTA_GROUP=2)
//   -H "#define MMA_N ..."  → 128/256
// Runtime args:
//   -0 iters, -1 sf_mode, -2 sign_mode

#ifndef MMA_M
#define MMA_M 128
#endif
#ifndef MMA_N
#define MMA_N 128
#endif
#ifndef K_SIZE
#define K_SIZE 0
#endif
#ifndef CTA_GROUP
#define CTA_GROUP 1
#endif
#if K_SIZE == 0
#define MMA_K 64
#else
#define MMA_K 96
#endif

#define SIGN_MASK 0x88888888u

extern "C" __global__ __launch_bounds__(32, 1)
#if CTA_GROUP == 2
__cluster_dims__(2, 1, 1)
#endif
void kernel(float* A, float* B, float* C, int iters, int sf_mode, int sign_mode) {
    __shared__ __align__(1024) unsigned smem_A[3072];
    __shared__ __align__(1024) unsigned smem_B[3072];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    int smem_size = MMA_K * 16;

    // Fill A randomly
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            smem_A[idx] = 0xDEADBEEFu ^ idx * 0xCAFEBABEu;
        }
    }
    // Fill B with controlled sign
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            int k = idx / 16;
            int npack = idx % 16;
            int n_base = npack * 8;
            unsigned r = 0xDEADBEEFu ^ idx * 0x13579BDFu;
            unsigned sign_set = 0;
            switch (sign_mode) {
                case 0: sign_set = (r >> 3) & 0xFF; break;
                case 1: sign_set = 0x00; break;
                case 2: sign_set = 0xFF; break;
                case 31: sign_set = 0xAA; break;
                case 3: { unsigned r0 = 0xDEADBEEFu ^ npack * 0x13579BDFu;
                          sign_set = (r0 >> 3) & 0xFF; break; }
                case 4: { unsigned r0 = 0xDEADBEEFu ^ k * 0x13579BDFu;
                          sign_set = ((r0>>15)&1) ? 0xFF : 0x00; break; }
                default: sign_set = (r >> 3) & 0xFF;
            }
            unsigned sign_bits = 0;
            for (int p = 0; p < 8; p++) if ((sign_set>>p)&1) sign_bits |= (1u<<(4*p+3));
            smem_B[idx] = (r & ~SIGN_MASK) | sign_bits;
        }
    }
    if (threadIdx.x == 0) {
        tmem_slot = 0xFFFFFFFFu;
        // FIX: commit.cta_group::2.mbarrier::arrive::one arrives ONCE only on leader's mbarrier
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)));
    }
    __syncthreads();
#if CTA_GROUP == 2
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
#endif

#if CTA_GROUP == 1
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
#else
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
#endif
    __syncthreads();
    unsigned tmem_addr = tmem_slot;
    unsigned tsfa_addr = tmem_addr + 128;
    unsigned tsfb_addr = tmem_addr + 256;

    // Initialize SFA/SFB based on sf_mode
    // sf_mode 0 = SF=1.0 (UE4M3 byte 0x38), 1 = SF=0, 2 = SF random
    {
        unsigned sf_pack;
        if (sf_mode == 0) sf_pack = 0x38383838u;
        else if (sf_mode == 1) sf_pack = 0x00000000u;
        else sf_pack = 0xDEADBEEFu;  // pseudo-random base; varies per write below
        for (int chunk = 0; chunk < 4; chunk++) {
            unsigned col_base = chunk * 128 + (threadIdx.x * 4);
            unsigned addr = tmem_addr + col_base;
            unsigned packs[4];
            if (sf_mode == 2) {
                // randomize each write
                for (int i = 0; i < 4; i++) {
                    unsigned x = (chunk * 128 + threadIdx.x * 4 + i) * 0x9E3779B1u + 0xCAFEBABEu;
                    x ^= x >> 16; x *= 0xCAFEBABEu;
                    // limit exponent to non-extreme to avoid Inf/NaN in UE4M3
                    unsigned char b0 = (x      ) & 0xFF; if ((b0&0x7F)>0x7E||(b0&0x7F)==0) b0=0x38;
                    unsigned char b1 = (x >>  8) & 0xFF; if ((b1&0x7F)>0x7E||(b1&0x7F)==0) b1=0x38;
                    unsigned char b2 = (x >> 16) & 0xFF; if ((b2&0x7F)>0x7E||(b2&0x7F)==0) b2=0x38;
                    unsigned char b3 = (x >> 24) & 0xFF; if ((b3&0x7F)>0x7E||(b3&0x7F)==0) b3=0x38;
                    packs[i] = b0 | (b1<<8) | (b2<<16) | (b3<<24);
                }
            } else {
                for (int i = 0; i < 4; i++) packs[i] = sf_pack;
            }
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};\n"
                :: "r"(addr), "r"(packs[0]), "r"(packs[1]), "r"(packs[2]), "r"(packs[3]));
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
#if CTA_GROUP == 1
    unsigned long long SBO = 2 * (unsigned long long)MMA_M;
#else
    // 2-CTA: per-CTA chunk is half of MMA_M
    unsigned long long SBO = 2 * (unsigned long long)(MMA_M / 2);
#endif
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);

    unsigned long long t0=0, t1=0;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

#if CTA_GROUP == 1
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
#else
    // 2-CTA: only leader CTA (every even block) issues MMA + commit
    if ((blockIdx.x % 2) == 0 && threadIdx.x == 0) {
        unsigned scaleC = 0;
        for (int i = 0; i < iters; i++) {
            asm volatile(
                "{\n\t .reg .pred P;\n\t setp.ne.b32 P, %4, 0;\n\t"
                "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], P;\n\t}"
                :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                   "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr) : "memory");
            scaleC = 1;
        }
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
    }
    // Only LEADER waits on mbarrier; non-leader just sync via cluster barrier
    if ((blockIdx.x % 2) == 0 && threadIdx.x == 0) {
        unsigned phase = 0;
        asm volatile(
            "{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase));
    }
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    if ((blockIdx.x % 2) == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
#endif
#if CTA_GROUP == 1
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
#else
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
#endif

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        printf("NVFP4 K=%d M=%d N=%d cta=%d sf=%d sign=%d cy/MMA=%.2f\n",
               MMA_K, MMA_M, MMA_N, CTA_GROUP, sf_mode, sign_mode, (double)(t1-t0)/iters);
    }
}
