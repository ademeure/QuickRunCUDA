// tcgen05.mma BF16 power test: B with various replication patterns.
// A is always random. SF unused (BF16 path).
// B layout: K=16 × N=128 BF16, in smem_B[1024] unsigned (each unsigned = 2 BF16).
//
// REP_MODE selects replication structure:
//   0 = pure random B (baseline)
//   1 = K-pair (every 2 K-rows identical: K[2i] == K[2i+1])
//   2 = K-quad (every 4 K-rows identical: K[4i..4i+3] same)
//   3 = K-half (K[0..7] == K[8..15])
//   4 = K-all (all 16 K-rows identical)
//   5 = N-pair (N[2i] == N[2i+1])
//   6 = N-quad
//   7 = N-stride 8
//   8 = N-stride 16
//   9 = N-stride 32
//   10 = N-stride 64
//   11 = N-stride 128 (whole N row identical)
//   12 = both K and N halved (combined replication)

#define MMA_M 128
#define MMA_N 128
#define MMA_K 16
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

    // B with replication pattern
    // B is K=16 rows × N=128 cols BF16. Layout: smem_B[k * (N/2) + n/2] holds n_even and n_odd packed.
    // Total = 16 * 64 unsigned = 1024 unsigned.
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < 1024; idx += 32) {
            int k = idx / 64;          // 0..15
            int npair = idx % 64;      // 0..63 (each pair of N)
            int n = npair * 2;         // 0,2,...,126
            // Apply replication
            int k_src = k, n_src = n;
            switch (rep_mode) {
                case 0: break;                                    // pure random
                case 1: k_src = k & ~1; break;                    // K-pair
                case 2: k_src = k & ~3; break;                    // K-quad
                case 3: k_src = k & ~7; break;                    // K-half
                case 4: k_src = 0; break;                         // K-all
                case 5: n_src = n & ~1; break;                    // N-pair (each N pair same)
                case 6: n_src = n & ~3; break;                    // N-quad (4 N's same)
                case 7: n_src = n & ~7; break;                    // N-stride 8
                case 8: n_src = n & ~15; break;                   // N-stride 16
                case 9: n_src = n & ~31; break;                   // N-stride 32
                case 10: n_src = n & ~63; break;                  // N-stride 64
                case 11: n_src = 0; break;                        // N-all
                case 12: k_src = k & ~7; n_src = n & ~63; break;  // both halved
            }
            // Random for the SOURCE position
            unsigned src_idx = (k_src * 64) + (n_src / 2);
            unsigned w = 0xDEADBEEFu ^ src_idx * 0x13579BDFu;
            // For modes that don't change n (K-mode), each pair of N is naturally distinct in w
            // For N-modes that group N: ensure both halves of unsigned word follow same source
            // when n_src is forced even, low half = high half within same group
            // For simplicity: always store from src_idx so packed word represents source
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

    unsigned idesc = (1U << 4) | (1U << 7) | (1U << 10)   // c=F32, A=BF16, B=BF16
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
        printf("BF16 brep rep_mode=%d iters=%d cy/MMA=%.2f\n",
               rep_mode, iters, (double)(t1-t0)/iters);
    }
}
