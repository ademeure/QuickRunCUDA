// FP8 e4m3 K-vary vs N-vary B power test (analog of BF16 K-vary).
// Tests if FP8 multiplier has same broadcast-A / distributed-B structure.
// FP8 e4m3: 1 sign + 4 exp + 3 mant = 8 bits. K=32 per MMA inst.
//
// Mode 200: random A & B
// Mode 300: B all-zero
// Mode 1400: B all = 0x38 (FP8 e4m3 +1.0)
// Mode 1600..1610: B K-vary K_unique = 1, 2, 4, ..., 1024 (cap at table size 16)
// Mode 1500..1510: B N-vary K_unique = same
// Mode 1700: A all = 0x38
// Mode 1701..1710: A K-vary

#define MMA_M 128
#define MMA_N 128
#define MMA_K 32

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int iters, int mode, int verify) {
    __shared__ __align__(1024) unsigned smem_A[2048];
    __shared__ __align__(1024) unsigned smem_B[2048];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    // FP8 val table (each byte = one FP8 e4m3 value)
    auto fp8_val = [](int v) -> unsigned char {
        // Various small normal FP8 e4m3 values (approximations)
        static const unsigned char vals[16] = {
            0x38, 0x40, 0x48, 0x3C, 0xB8, 0x44, 0x50, 0x30,
            0x4C, 0xBC, 0x54, 0x34, 0x58, 0xC8, 0x5C, 0x3E
        };
        return vals[v & 15];
    };

    // A pattern
    if (threadIdx.x < 32) {
        for (int i = 0; i < 64; i++) {
            unsigned idx = threadIdx.x + i*32;
            unsigned r_a = 0xDEADBEEFu ^ idx * 0xCAFEBABEu;
            unsigned w_a;
            if (mode == 1700) {
                w_a = 0x38383838u;  // A const +1.0
            } else if (mode >= 1701 && mode <= 1710) {
                // FIXED: A is M=128 × K=32 FP8 = 1024 unsigned (4 FP8 per word)
                // smem_A[m*8 + k_pack], k_pack = 0..7, each pack holds 4 K values
                // So per-cycle K index changes faster, varies as k_pack within each m row
                int K_unique = 1 << (mode - 1700);
                int k_pack = idx % 8;  // K-pack index 0..7
                int n_in_pack = 0;     // simplification - all 4 K positions in pack get same value
                int k_eff = k_pack * 4 + n_in_pack;
                int v_idx = (k_eff & (K_unique - 1));
                unsigned char v = fp8_val(v_idx);
                // Make all 4 FP8 in word same value (since they're 4 consecutive K positions
                // in same K-pack — but K-vary should distinguish K positions)
                // For TRUE K-vary, want each FP8 in pack different
                if (K_unique <= 4) {
                    // pack the K_unique cycle within the word
                    unsigned char b0 = fp8_val((k_pack*4 + 0) & (K_unique - 1));
                    unsigned char b1 = fp8_val((k_pack*4 + 1) & (K_unique - 1));
                    unsigned char b2 = fp8_val((k_pack*4 + 2) & (K_unique - 1));
                    unsigned char b3 = fp8_val((k_pack*4 + 3) & (K_unique - 1));
                    w_a = b0 | (b1<<8) | (b2<<16) | (b3<<24);
                } else {
                    unsigned char b0 = fp8_val((k_pack*4 + 0) & (K_unique - 1));
                    unsigned char b1 = fp8_val((k_pack*4 + 1) & (K_unique - 1));
                    unsigned char b2 = fp8_val((k_pack*4 + 2) & (K_unique - 1));
                    unsigned char b3 = fp8_val((k_pack*4 + 3) & (K_unique - 1));
                    w_a = b0 | (b1<<8) | (b2<<16) | (b3<<24);
                }
            } else {
                w_a = r_a;
            }
            smem_A[idx] = w_a;
        }
    }

    // B pattern
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < 1024; idx += 32) {
            unsigned r = 0xDEADBEEFu ^ idx * 0x13579BDFu;
            unsigned w;
            if (mode == 200) {
                w = r;
            } else if (mode == 300) {
                w = 0;
            } else if (mode == 1400) {
                w = 0x38383838u;
            } else if (mode >= 1500 && mode <= 1510) {
                // N-vary
                int K_unique = 1 << (mode - 1500);
                // Layout: smem_B[k * (N/4) + n_pack] for FP8 K=32 N=128 (32*128/4 = 1024 unsigned)
                int n_pack = idx % 32;  // 0..31
                int n_base = n_pack * 4;
                unsigned char b0 = fp8_val((n_base + 0) % K_unique);
                unsigned char b1 = fp8_val((n_base + 1) % K_unique);
                unsigned char b2 = fp8_val((n_base + 2) % K_unique);
                unsigned char b3 = fp8_val((n_base + 3) % K_unique);
                w = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            } else if (mode >= 1600 && mode <= 1610) {
                // K-vary
                int K_unique = 1 << (mode - 1600);
                int k = idx / 32;
                unsigned char v = fp8_val(k % K_unique);
                w = v | (v << 8) | (v << 16) | (v << 24);
            } else {
                w = r;
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

    if (verify && threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < 2; i++) {
            unsigned w = smem_B[i];
            printf("  B[%d] = 0x%08x  bytes %02x %02x %02x %02x\n", i, w,
                   w & 0xFF, (w>>8) & 0xFF, (w>>16) & 0xFF, (w>>24) & 0xFF);
        }
    }

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    __syncthreads();
    unsigned tmem_addr = tmem_slot;

    // FP8 e4m3: kind::f8f6f4, a_format=0, b_format=0
    unsigned idesc = (1U << 4)
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
        printf("FP8 mode=%d iters=%d cy/MMA=%.2f\n", mode, iters, (double)(t1-t0)/iters);
    }
}
