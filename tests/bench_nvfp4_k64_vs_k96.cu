// K=64 vs K=96 ULTRA NVFP4 throughput comparison via raw PTX.
//
// Identical M, N, smem layout, scale factor population.
// Only differences:
//   - MMA_K (smem_B size)
//   - idesc bit 31 (scale_vec field): 0 = scale_vec::4X (K=64), 1 = scale_vec::1X (K=96)
//   - PTX block_scale form is "block16" in both cases (block-size-16 SF granularity)
//
// Per spec / catalog §49:
//   - K=64 standard ("scale_vec::4X") gives ~10 PF chip-wide
//   - K=96 ULTRA  ("scale_vec::1X") gives ~15 PF chip-wide (1.5x density per inst)
//
// Args:
//   -0 (iters)  : MMA iterations per kernel
//   -1 (mode)   : 64 -> K=64 standard, 96 -> K=96 ULTRA
//   -2 unused
//
// Geometry: 2-CTA cluster (M_total=256, N=256), 32 threads/CTA, 2 blocks total per cluster.

#define MMA_M 256
#define MMA_N 256

extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void kernel(float* A, float* B, float* C, int iters, int mode, int u2) {
    // Allocate biggest case (K=96 ⇒ 3072 dwords)
    __shared__ __align__(1024) unsigned smem_A[3072];
    __shared__ __align__(1024) unsigned smem_B[3072];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    const int MMA_K = (mode == 96) ? 96 : 64;
    int smem_size = MMA_K * MMA_N / 8;

    // Random A and B (unsigned packed nibble lanes)
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            unsigned r = (idx + blockIdx.x * 1024u) * 0x9E3779B1u;
            r ^= r >> 16; r *= 0x85EBCA6Bu;
            r ^= r >> 13; r *= 0xC2B2AE35u;
            r ^= r >> 16;
            smem_A[idx] = r;
        }
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            unsigned r = (idx + blockIdx.x * 1024u + 0xC0FFEE00u) * 0x9E3779B1u;
            r ^= r >> 16; r *= 0x85EBCA6Bu;
            r ^= r >> 13; r *= 0xC2B2AE35u;
            r ^= r >> 16;
            smem_B[idx] = r;
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

    // Init TMEM SF region: UE4M3 1.0 (byte 0x38)
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

    // idesc: a_format=E2M1(5)<<7 | b_format=E2M1(5)<<10
    //        | n_dim = N/8 << 17 | m_dim = M/16 << 24
    //        | scale_vec bit 31: 0=4X(K=64) / 1=1X(K=96 ULTRA)
    unsigned idesc_base = (5U << 7) | (5U << 10)
                        | (((unsigned)MMA_N >> 3) << 17)
                        | (((unsigned)MMA_M >> 4) << 24);
    unsigned idesc_k64  = idesc_base;
    unsigned idesc_k96  = idesc_base | (1U << 31);

    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16;
    unsigned long long SBO = 2 * (unsigned long long)(MMA_M / 2);
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);

    unsigned long long t0 = 0, t1 = 0;
    if ((blockIdx.x % 2) == 0 && threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned scaleC = 0;
        if (mode == 96) {
            // K=96 ULTRA path
            for (int i = 0; i < iters; i++) {
                asm volatile(
                    "{\n\t .reg .pred PRED;\n\t setp.ne.b32 PRED, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], PRED;\n\t}"
                    :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc_k96),
                       "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr) : "memory");
                scaleC = 1;
            }
        } else {
            // K=64 standard path
            for (int i = 0; i < iters; i++) {
                asm volatile(
                    "{\n\t .reg .pred PRED;\n\t setp.ne.b32 PRED, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], PRED;\n\t}"
                    :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc_k64),
                       "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr) : "memory");
                scaleC = 1;
            }
        }
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
        unsigned phase_w = 0;
        asm volatile("{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t @P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase_w));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));

    if ((blockIdx.x % 2) == 0 && threadIdx.x == 0) {
        ((unsigned long long*)C)[blockIdx.x / 2] = t1 - t0;
        if (blockIdx.x == 0) {
            printf("MODE=%d (K=%d) iters=%d cy/MMA=%.2f total_cy=%llu\n",
                mode, MMA_K, iters, (double)(t1 - t0) / iters, t1 - t0);
        }
    }
}
