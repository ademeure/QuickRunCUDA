// NVFP4 K=96, B has 5-positive distribution {+0..+2.0} but with 1 OUTLIER per
// K-block-of-16 elements (i.e. per SF block). Outlier picked uniformly from
// {-3, -2, -1, +2, +3, +4} = codes {0xD, 0xC, 0xA, 0x4, 0x5, 0x6}.
// Position of outlier within block-of-16 is random per (n, kblock_id).
// SF mode: u1 = 0 (SF=1.0) or 1 (SF random byte).
// Args: u0 = iters, u1 = sf_mode, u2 = unused
#define MMA_M 256
#define MMA_N 256
#define MMA_K 96   // 6 K-blocks of 16

__device__ __forceinline__ unsigned mix32(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void kernel(float* A, float* B, float* C, int iters, int sf_mode, int u2) {
    __shared__ __align__(1024) unsigned smem_A[3072];
    __shared__ __align__(1024) unsigned smem_B[3072];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    int smem_size = MMA_K * MMA_N / 8;       // 3072 packed dwords
    int npacks = MMA_N / 8;                  // 32 packs per K-row

    // A: fully random
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            unsigned r = (idx + blockIdx.x * 1024u) * 0x9E3779B1u;
            r ^= r >> 16; r *= 0x85EBCA6Bu;
            r ^= r >> 13; r *= 0xC2B2AE35u;
            r ^= r >> 16;
            smem_A[idx] = r;
        }
    }

    // B: 5-pos {+0..+2} with 1 outlier per (n, k-block) from {-3,-2,-1,+2,+3,+4}
    // Outlier codes: 0xD=-3, 0xC=-2, 0xA=-1, 0x4=+2, 0x5=+3, 0x6=+4
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            int k = idx / npacks;
            int npack = idx % npacks;
            int n_base = npack * 8;
            int kblock = k / 16;
            int k_in_block = k & 15;
            // Per-element random
            unsigned r_base = mix32((unsigned)idx + blockIdx.x * 1024u + 0xC0FFEE00u);
            unsigned val = 0;
            for (int p = 0; p < 8; p++) {
                int n = n_base + p;
                // Outlier position for this (n, kblock): pick deterministically
                unsigned out_h = mix32((unsigned)n * 0x9E3779B1u + (unsigned)kblock * 0xCAFEBABEu);
                int outlier_k = (int)(out_h & 15);     // 0..15
                bool is_outlier = (k_in_block == outlier_k);
                unsigned base = (r_base >> (p * 4)) & 0xFFu;
                unsigned fp4;
                if (is_outlier) {
                    // pick uniform from 6 outlier codes
                    static const unsigned char outliers[6] = {0xD, 0xC, 0xA, 0x4, 0x5, 0x6};
                    fp4 = outliers[base % 6u];
                } else {
                    fp4 = base % 5u;                   // 0x0..0x4 = 5-pos set
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
        for (int chunk = 0; chunk < 4; chunk++) {
            unsigned col_base = chunk * 128 + (threadIdx.x * 4);
            unsigned addr = tmem_addr + col_base;
            unsigned pa, pb, pc, pd;
            if (sf_mode == 0) {
                pa = pb = pc = pd = 0x38383838u;
            } else {
                unsigned seed = (chunk * 32u + threadIdx.x) * 0x9E3779B1u + 0xCAFEFACEu;
                seed = mix32(seed);
                pa = mix32(seed + 1u); pb = mix32(seed + 2u);
                pc = mix32(seed + 3u); pd = mix32(seed + 4u);
            }
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};\n"
                :: "r"(addr), "r"(pa), "r"(pb), "r"(pc), "r"(pd));
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
