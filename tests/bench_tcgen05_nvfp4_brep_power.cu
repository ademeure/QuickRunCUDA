// NVFP4 (kind::mxf4nvf4.block_scale.block16) B-replication power sweep
// m=128 n=128 K=64. SF=1.0 always. A always random.
// REP_MODE selects N-direction replication stride (0=baseline rand)

#define MMA_M 128
#define MMA_N 128
#define MMA_K 64
#ifndef REP_MODE
#define REP_MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int iters, int rep_mode, int u2) {
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

    // B layout: K=64 × N=128 FP4 = 8KB. Each unsigned = 8 FP4 (8 N elements).
    // smem_B[k * (N/8) + n/8] holds N[n..n+7]
    // Total = 64 * 16 = 1024 unsigned
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < 1024; idx += 32) {
            int k = idx / 16;
            int npack = idx % 16;       // 0..15 (each pack of 8 N)
            // Apply N-stride replication
            int stride;
            switch (rep_mode) {
                case 0: stride = 1; break;     // pure random (no replication)
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
            // For N-stride S: pack N[i..i+7] all share same value if S>=8.
            // Otherwise the pack contains S different values repeated 8/S times.
            unsigned w;
            if (stride >= 8) {
                // Whole pack uses same source: round npack down to (stride/8) boundary
                int npack_grp = stride / 8;     // # packs sharing same value
                int npack_src = (npack / npack_grp) * npack_grp;
                unsigned src = (unsigned)(k * 16 + npack_src);
                w = 0xDEADBEEFu ^ src * 0x13579BDFu;
            } else {
                // Within pack: stride=1/2/4 → some FP4 nibbles repeat
                unsigned base = (unsigned)(k * 16 + npack);
                unsigned r = 0xDEADBEEFu ^ base * 0x13579BDFu;
                if (stride == 1) {
                    w = r;
                } else if (stride == 2) {
                    // Pairs of nibbles same: bytes 0xXY where Y=X
                    w = ((r & 0xF0F0F0F0u) >> 4) | (r & 0xF0F0F0F0u);
                } else if (stride == 4) {
                    // Each nibble pair → use same byte pattern across word repeated
                    unsigned b0 = r & 0xFF;
                    w = (b0 << 24) | (b0 << 16) | (b0 << 8) | b0;
                } else {
                    w = r;
                }
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
    unsigned tsfa_addr = tmem_addr + 128;
    unsigned tsfb_addr = tmem_addr + 256;

    // Initialize entire TMEM with UE4M3 1.0 (0x38)
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

    unsigned idesc = (5U << 7) | (5U << 10)            // a=B=E2M1
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
        printf("NVFP4 brep rep_mode=%d iters=%d cy/MMA=%.2f\n",
               rep_mode, iters, (double)(t1-t0)/iters);
    }
}
