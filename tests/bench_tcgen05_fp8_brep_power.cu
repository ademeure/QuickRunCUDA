// FP8 e4m3 N-stride B replication power sweep
// Same methodology as BF16 brep, but kind::f8f6f4 instead of kind::f16

#define MMA_M 128
#define MMA_N 128
#define MMA_K 32   // FP8 K=32
#ifndef REP_MODE
#define REP_MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int iters, int rep_mode, int u2) {
    __shared__ __align__(1024) unsigned smem_A[2048];
    __shared__ __align__(1024) unsigned smem_B[2048];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    if (threadIdx.x < 32) {
        for (int i = 0; i < 64; i++) {
            unsigned idx = threadIdx.x + i*32;
            smem_A[idx] = 0xDEADBEEFu ^ idx * 0xCAFEBABEu;
        }
    }
    // B = K=32 × N=128 FP8e4m3 = 4KB. Each unsigned = 4 FP8 (4 N elements)
    // smem_B[k * (N/4) + n/4] holds N[n..n+3]
    // Total = 32 * 32 = 1024 unsigned
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < 1024; idx += 32) {
            int k = idx / 32;
            int npack = idx % 32;       // 0..31 (each pack of 4 N)
            int stride;
            switch (rep_mode) {
                case 0: stride = 1; break;
                case 1: stride = 2; break;
                case 2: stride = 4; break;
                case 3: stride = 8; break;
                case 4: stride = 16; break;
                case 5: stride = 24; break;
                case 6: stride = 32; break;
                case 7: stride = 48; break;
                case 8: stride = 64; break;
                case 9: stride = 96; break;
                case 10: stride = 128; break;
                default: stride = 1; break;
            }
            unsigned w;
            if (stride >= 4) {
                int npack_grp = stride / 4;
                int npack_src = (npack / npack_grp) * npack_grp;
                unsigned src = (unsigned)(k * 32 + npack_src);
                w = 0xDEADBEEFu ^ src * 0x13579BDFu;
            } else {
                unsigned base = (unsigned)(k * 32 + npack);
                unsigned r = 0xDEADBEEFu ^ base * 0x13579BDFu;
                if (stride == 1) w = r;
                else if (stride == 2) {
                    // bytes 0,1 same; 2,3 same → pairs of N share FP8
                    unsigned b0 = r & 0xFF;
                    unsigned b2 = (r >> 16) & 0xFF;
                    w = (b2 << 24) | (b2 << 16) | (b0 << 8) | b0;
                } else w = r;
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

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    __syncthreads();
    unsigned tmem_addr = tmem_slot;

    // FP8 e4m3: a_format=0, b_format=0, kind=f8f6f4
    unsigned idesc = (1U << 4)                            // c=F32
                   | (0U << 7) | (0U << 10)               // a=b=E4M3
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
                "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%4, %5, %6, %7}, PRED;\n\t}"
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
        printf("FP8 brep rep_mode=%d iters=%d cy/MMA=%.2f\n",
               rep_mode, iters, (double)(t1-t0)/iters);
    }
}
