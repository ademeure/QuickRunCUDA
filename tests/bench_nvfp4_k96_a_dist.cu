// NVFP4 K=96 power vs A-side distribution.
// B is FIXED (mode-controlled), A is varied across modes.
//
// Args: u0=iters, u1=A_mode, u2=B_mode
// A_mode (u1) and B_mode (u2):
//   0 = constant 0x0 (all +0)
//   1 = constant 0x4 (all +2.0)
//   2 = 5 positive {+0..+2} (5-pos low-mag set)
//   3 = 8 positive (sign always 0, full mantissa)
//   4 = full random 16
#define MMA_M 256
#define MMA_N 256
#define MMA_K 96

__device__ __forceinline__ unsigned mix32(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ unsigned pick(unsigned byte, int mode) {
    if (mode == 0) return 0x0u;
    if (mode == 1) return 0x4u;
    if (mode == 2) return byte % 5u;
    if (mode == 3) return byte & 0x7u;
    if (mode == 4) return byte & 0xFu;
    return 0u;
}

extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void kernel(float* A, float* B, float* C, int iters, int a_mode, int b_mode) {
    __shared__ __align__(1024) unsigned smem_A[3072];
    __shared__ __align__(1024) unsigned smem_B[3072];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;
    int smem_size = MMA_K * MMA_N / 8;

    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            unsigned r = mix32((unsigned)idx + blockIdx.x * 1024u + 0xAAAA0000u);
            unsigned val = 0;
            for (int p = 0; p < 8; p++) {
                unsigned byte = (r >> (p * 4)) & 0xFFu;
                unsigned fp4 = pick(byte, a_mode) & 0xFu;
                val |= (fp4 << (p * 4));
            }
            smem_A[idx] = val;
        }
    }
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            unsigned r = mix32((unsigned)idx + blockIdx.x * 1024u + 0xC0FFEE00u);
            unsigned val = 0;
            for (int p = 0; p < 8; p++) {
                unsigned byte = (r >> (p * 4)) & 0xFFu;
                unsigned fp4 = pick(byte, b_mode) & 0xFu;
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
        unsigned one_pack = 0x38383838u;
        for (int chunk = 0; chunk < 4; chunk++) {
            unsigned col_base = chunk * 128 + (threadIdx.x * 4);
            unsigned addr = tmem_addr + col_base;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};\n"
                :: "r"(addr), "r"(one_pack), "r"(one_pack), "r"(one_pack), "r"(one_pack));
        }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;");
    __syncthreads();
    unsigned idesc = (5U << 7) | (5U << 10) | (((unsigned)MMA_N >> 3) << 17)
                   | (((unsigned)MMA_M >> 4) << 24) | (1U << 31);
    auto desc_encode = [](unsigned long long x) -> unsigned long long { return (x & 0x3FFFFULL) >> 4; };
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
        asm volatile("{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t @P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase_w));
    }
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
}
