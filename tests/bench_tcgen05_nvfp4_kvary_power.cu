// NVFP4 K-vary vs N-vary B power test (analog of BF16 K-vary).
// Tests if NVFP4 (kind::mxf4nvf4.block_scale) multiplier has same
// broadcast-A / distributed-B structure as BF16/FP8.
//
// NVFP4 e2m1: 4 bits per value (sign + 2exp + 1mant). K=64 per MMA inst.
// Each unsigned word holds 8 FP4 values.
// SF (scale factors) held at UE4M3 1.0 (byte 0x38).
//
// Mode 200: random A & B
// Mode 300: B all-zero
// Mode 1400: B all = 0x22 (FP4 +1.0 / +1.0 packed)
// Mode 1500..1510: B N-vary K_unique values (cap at 16)
// Mode 1600..1610: B K-vary K_unique values
// Mode 1700: A all = 0x22
// Mode 1701..1710: A K-vary K_unique values

#define MMA_M 128
#define MMA_N 128
#define MMA_K 64

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int iters, int mode, int verify) {
    __shared__ __align__(1024) unsigned smem_A[2048];
    __shared__ __align__(1024) unsigned smem_B[2048];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    // FP4 nibble val table - small normal/finite values
    auto fp4_nib = [](int v) -> unsigned char {
        // 4-bit FP4 e2m1 values (low nibble)
        // 0=+0, 1=+0.5, 2=+1.0, 3=+1.5, 4=+2.0, 5=+3.0, 6=+4.0, 7=+6.0
        // 8=-0, 9=-0.5, 10=-1.0, 11=-1.5, 12=-2.0, 13=-3.0, 14=-4.0, 15=-6.0
        static const unsigned char nibs[16] = {
            0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0xA, 0xB,
            0xC, 0xD, 0xE, 0xF, 0x1, 0x9, 0x2, 0x4
        };
        return nibs[v & 15];
    };

    // A: M=128 K=64 FP4 = 8192 FP4 = 1024 unsigned (8 FP4 per word)
    // Layout: smem_A[m*8 + k_pack], k_pack = 0..7, each pack holds 8 K positions
    if (threadIdx.x < 32) {
        for (int i = 0; i < 64; i++) {
            unsigned idx = threadIdx.x + i*32;
            unsigned r_a = 0xDEADBEEFu ^ idx * 0xCAFEBABEu;
            unsigned w_a;
            if (mode == 1700) {
                w_a = 0x22222222u;  // all FP4 +1.0
            } else if (mode >= 1701 && mode <= 1710) {
                int K_unique = 1 << (mode - 1700);
                int k_pack = idx % 8;
                // Each FP4 in word is at K position = k_pack*8 + p (p=0..7)
                w_a = 0;
                for (int p = 0; p < 8; p++) {
                    int k_pos = k_pack * 8 + p;
                    unsigned char nib = fp4_nib(k_pos & (K_unique - 1));
                    w_a |= ((unsigned)nib) << (4*p);
                }
            } else {
                w_a = r_a;
            }
            smem_A[idx] = w_a;
        }
    }

    // B: K=64 N=128 FP4 = 8192 FP4 = 1024 unsigned (8 FP4 per word)
    // Layout: smem_B[k * 16 + n_pack], n_pack = 0..15, each pack holds 8 N positions
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < 1024; idx += 32) {
            unsigned r = 0xDEADBEEFu ^ idx * 0x13579BDFu;
            unsigned w;
            if (mode == 200) {
                w = r;
            } else if (mode == 300) {
                w = 0;
            } else if (mode == 1400) {
                w = 0x22222222u;
            } else if (mode >= 1500 && mode <= 1510) {
                // N-vary
                int K_unique = 1 << (mode - 1500);
                int n_pack = idx % 16;
                w = 0;
                for (int p = 0; p < 8; p++) {
                    int n_pos = n_pack * 8 + p;
                    unsigned char nib = fp4_nib(n_pos & (K_unique - 1));
                    w |= ((unsigned)nib) << (4*p);
                }
            } else if (mode >= 1600 && mode <= 1610) {
                // K-vary: each FP4 within word can be different K position
                // BUT actually a single word covers 8 N positions at one K row
                // So all 8 FP4 in word should be same value (same K, different N)
                int K_unique = 1 << (mode - 1600);
                int k = idx / 16;
                unsigned char nib = fp4_nib(k & (K_unique - 1));
                unsigned char b = (nib << 4) | nib;  // both nibbles same in byte
                w = b | (b<<8) | (b<<16) | (b<<24);
            } else if (mode >= 2700 && mode <= 2710) {
                // PURE N-vary HIGH-ENTROPY (NVFP4): N_unique = 1<<(mode-2700) = 1..1024
                // K constant per row. NVFP4: 8 FP4 per word, 16 word_packs across N=128.
                int N_unique = 1 << (mode - 2700);
                int n_pack = idx % 16;
                w = 0;
                auto h = [](int nn) -> unsigned char {
                    unsigned hh = 0xC0FFEE13u ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned char)((hh & 0x07) | ((hh >> 4) & 0x08));  // any FP4 (0..15)
                };
                for (int p = 0; p < 8; p++) {
                    int n_pos = n_pack * 8 + p;
                    unsigned char nib = h(n_pos % N_unique);
                    w |= ((unsigned)(nib & 0x0F)) << (4*p);
                }
            } else if (mode >= 2750 && mode <= 2799) {
                // FINE NVFP4 N-vary: N_unique = mode-2750 (1..49)
                int N_unique = mode - 2750;
                if (N_unique < 1) N_unique = 1;
                int n_pack = idx % 16;
                w = 0;
                auto h = [](int nn) -> unsigned char {
                    unsigned hh = 0xC0FFEE13u ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned char)((hh & 0x07) | ((hh >> 4) & 0x08));
                };
                for (int p = 0; p < 8; p++) {
                    int n_pos = n_pack * 8 + p;
                    unsigned char nib = h(n_pos % N_unique);
                    w |= ((unsigned)(nib & 0x0F)) << (4*p);
                }
            } else if (mode >= 5200 && mode <= 5264) {
                // NVFP4 K-row CONSECUTIVE GROUPING: K_unique = mode-5200, group_size=64/K_unique
                int K_unique = mode - 5200;
                if (K_unique < 1) K_unique = 1;
                if (K_unique > 64) K_unique = 64;
                int group_size = 64 / K_unique;
                if (group_size < 1) group_size = 1;
                int n_pack = idx % 16;
                int k = idx / 16;
                int k_group = k / group_size;
                w = 0;
                auto h = [](int kk, int nn) -> unsigned char {
                    unsigned hh = 0xC0FFEE13u ^ (kk * 0xDEADBEEFu) ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned char)((hh & 0x07) | ((hh >> 4) & 0x08));
                };
                // Within-row N_unique=64 (1 sub-tile only since NVFP4 sub-tile = 64 N)
                for (int p = 0; p < 8; p++) {
                    int n_pos = n_pack * 8 + p;
                    int n_idx = n_pos % 64;
                    unsigned char nib = h(k_group, n_idx);
                    w |= ((unsigned)(nib & 0x0F)) << (4*p);
                }
            } else if (mode >= 3020 && mode <= 3027) {
                // NVFP4 SINGLE UNIQUE POSITION at sub-tile P (8 pseudo-sub-tiles)
                int P = mode - 3020;
                int n_pack = idx % 16;
                int sub_tile = n_pack / 2;
                int pos_in_tile = n_pack % 2;
                int pattern_id = (sub_tile == P) ? (P + 1) : 0;
                w = 0;
                auto h = [](int nn) -> unsigned char {
                    unsigned hh = 0xC0FFEE13u ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned char)((hh & 0x07) | ((hh >> 4) & 0x08));
                };
                for (int p = 0; p < 8; p++) {
                    int n_in_tile = pos_in_tile * 8 + p;
                    unsigned char nib = h(n_in_tile + pattern_id * 100);
                    w |= ((unsigned)(nib & 0x0F)) << (4*p);
                }
            } else if (mode >= 3100 && mode <= 3108) {
                // PATTERN COUNT for NVFP4: rotating distinct sub-tile patterns 1..8
                int N_distinct = mode - 3100;
                if (N_distinct < 1) N_distinct = 1;
                if (N_distinct > 8) N_distinct = 8;
                int n_pack = idx % 16;
                int sub_tile = n_pack / 2;
                int pos_in_tile = n_pack % 2;
                int pattern_id = sub_tile % N_distinct;
                w = 0;
                auto h = [](int nn) -> unsigned char {
                    unsigned hh = 0xC0FFEE13u ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned char)((hh & 0x07) | ((hh >> 4) & 0x08));
                };
                for (int p = 0; p < 8; p++) {
                    int n_in_tile = pos_in_tile * 8 + p;
                    unsigned char nib = h(n_in_tile + pattern_id * 100);
                    w |= ((unsigned)(nib & 0x0F)) << (4*p);
                }
            } else if (mode >= 2900 && mode <= 2908) {
                // SUB-TILE BREAKING for NVFP4: 8 sub-tiles of 16 N each (= 2 word_packs each).
                // K_break unique sub-tiles, others share pattern 0
                int K_break = mode - 2900;
                int n_pack = idx % 16;
                int sub_tile = n_pack / 2;       // 8 sub-tiles, each = 2 word_packs (16 N values)
                int pos_in_tile = n_pack % 2;
                int pattern_id = (sub_tile < (8 - K_break)) ? 0 : sub_tile;
                w = 0;
                auto h = [](int nn) -> unsigned char {
                    unsigned hh = 0xC0FFEE13u ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned char)((hh & 0x07) | ((hh >> 4) & 0x08));
                };
                for (int p = 0; p < 8; p++) {
                    int n_in_tile = pos_in_tile * 8 + p;
                    unsigned char nib = h(n_in_tile + pattern_id * 100);
                    w |= ((unsigned)(nib & 0x0F)) << (4*p);
                }
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

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    __syncthreads();
    unsigned tmem_addr = tmem_slot;
    unsigned tsfa_addr = tmem_addr + 128;
    unsigned tsfb_addr = tmem_addr + 256;

    // Init TMEM SF region with UE4M3 1.0 (0x38)
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

    // NVFP4 idesc
    unsigned idesc = (5U << 7) | (5U << 10)
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
        printf("NVFP4 mode=%d iters=%d cy/MMA=%.2f\n", mode, iters, (double)(t1-t0)/iters);
    }
}
