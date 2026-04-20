// BF16 per-bit decomposition: force ONE bit position constant, others random.
// BF16 layout: bit 15=sign, bits 14:7=exp[7:0], bits 6:0=mant[6:0].
// Mode = 0..15 selects which B bit to force (constant 0).
// Mode = 100 + i: force B bit i constant 1.
// Mode = 200: B random (baseline).
// Mode = 300: B all-zero.
// Mode = 400 + i: force A bit i to 0 (B always random)
// Mode = 500 + i: force A bit i to 1 (B always random)
// Mode = 600..615: force B bits 0..i (cumulative low-to-high, 0=just bit 0, 15=all bits)
// Mode = 700..715: force B bits 15..(15-i) (cumulative high-to-low, 0=just sign, 15=all)
// Mode = 800: force B mantissa only (bits 0-6)
// Mode = 801: force B exp only (bits 7-14)
// Mode = 802: force B mant+exp (bits 0-14)
// Mode = 803: force B sign only (= mode 15)
// Mode = 804: force B sign+exp (bits 7-15)
// Mode = 805: force B sign+mant (bits 0-6,15)
//
// VERIFICATION: at startup, thread 0 of block 0 prints first 4 BF16 values
// of B (hex + decoded sign/exp/mant) so we can sanity-check encoding.

#define MMA_M 128
#define MMA_N 128
#define MMA_K 16
#ifndef MODE
#define MODE 200
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int iters, int mode, int verify) {
    __shared__ __align__(1024) unsigned smem_A[2048];
    __shared__ __align__(1024) unsigned smem_B[2048];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    // A pattern: random bytes, then force one bit position if mode 400+
    if (threadIdx.x < 32) {
        for (int i = 0; i < 64; i++) {
            unsigned idx = threadIdx.x + i*32;
            unsigned r_a = 0xDEADBEEFu ^ idx * 0xCAFEBABEu;
            unsigned w_a;
            if (mode >= 400 && mode <= 415) {
                int b = mode - 400;
                unsigned bm = (1u << b) | (1u << (b + 16));
                w_a = r_a & ~bm;
            } else if (mode >= 500 && mode <= 515) {
                int b = mode - 500;
                unsigned bm = (1u << b) | (1u << (b + 16));
                w_a = (r_a & ~bm) | bm;
            } else if (mode == 1300) {
                w_a = 0x7F807F80u;  // A = +Inf
            } else if (mode == 1303) {
                w_a = 0x7FFF7FFFu;  // A = NaN max-mant
            } else if (mode == 1305) {
                // A = NaN random mant (force exp=255, mant non-zero)
                w_a = (r_a & 0x007F007Fu) | 0x7F807F80u | 0x00010001u;
            } else if (mode == 1700) {
                // A = constant +1.0
                w_a = 0x3F803F80u;
            } else if (mode >= 1701 && mode <= 1710) {
                // A K-vary: K_unique values across K dim, same across M (broadcast test)
                int K_unique_k = 1 << (mode - 1700);  // 2,4,8,...
                int k = idx / 64;        // K index 0..15
                int v_idx = k % K_unique_k;
                auto val = [](int v) -> unsigned short {
                    static const unsigned short vals[16] = {
                        0x3F80, 0x4000, 0x40C0, 0x3FC0, 0xBF80, 0x4040, 0x4180, 0x3F00,
                        0x4080, 0xBFC0, 0x4100, 0x3F40, 0x4200, 0xC080, 0x4140, 0x3FE0
                    };
                    return vals[v & 15];
                };
                unsigned short v = val(v_idx);
                w_a = ((unsigned)v << 16) | v;
            } else if (mode >= 4000 && mode <= 4007) {
                // A M-vary HIGH-ENTROPY: M_unique = 1<<(mode-4000); B forced to constant +1.0
                // A layout: idx = m * 8 + k_pack. m = idx/8 (0..127), k_pack = idx%8.
                // For each k position, A[m,k] uses h(m % M_unique).
                int M_unique = 1 << (mode - 4000);
                int m_idx = (idx / 8) % M_unique;
                auto h = [](int mm) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (mm * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                unsigned short v = h(m_idx);
                w_a = ((unsigned)v << 16) | v;
            } else if (mode >= 4100 && mode <= 4107) {
                // A K-vary HIGH-ENTROPY: K_unique = 1<<(mode-4100). M_unique = 128 (each m row may differ but K-side determines)
                // For each m, K-side cycles through K_unique values
                int K_unique = 1 << (mode - 4100);
                int k = idx / 64;       // K row 0..15 (idx 0..63 maps to K=0)
                int v_idx = k % K_unique;
                auto h = [](int kk) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (kk * 0xDEADBEEFu);
                    hh = hh * 0x9E3779B1u;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                unsigned short v = h(v_idx);
                w_a = ((unsigned)v << 16) | v;
            } else if (mode >= 4200 && mode <= 4207) {
                // A FULL random per (m, k): truly random per byte position
                // M_unique = 1<<(mode-4200), all M cycles through this many distinct values; K varies fully
                int M_unique = 1 << (mode - 4200);
                int m_idx = (idx / 8) % M_unique;
                int k_pack = idx % 8;
                auto h = [](int mm, int kk) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (mm * 0x9E3779B1u) ^ (kk * 0xDEADBEEFu);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                unsigned short ve = h(m_idx, k_pack * 2);
                unsigned short vo = h(m_idx, k_pack * 2 + 1);
                w_a = ((unsigned)vo << 16) | ve;
            } else if (mode >= 1801 && mode <= 1810) {
                // A N-vary: K_unique values across M dim (no effect on per-MAC temporal)
                int K_unique_m = 1 << (mode - 1800);
                int npair = idx % 64;    // pretending m varies along this dim
                int v_idx = npair % K_unique_m;
                auto val = [](int v) -> unsigned short {
                    static const unsigned short vals[16] = {
                        0x3F80, 0x4000, 0x40C0, 0x3FC0, 0xBF80, 0x4040, 0x4180, 0x3F00,
                        0x4080, 0xBFC0, 0x4100, 0x3F40, 0x4200, 0xC080, 0x4140, 0x3FE0
                    };
                    return vals[v & 15];
                };
                unsigned short v = val(v_idx);
                w_a = ((unsigned)v << 16) | v;
            } else {
                w_a = r_a;
            }
            smem_A[idx] = w_a;
        }
    }

    // B pattern: random bytes, then force one bit position
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < 1024; idx += 32) {
            unsigned r = 0xDEADBEEFu ^ idx * 0x13579BDFu;
            unsigned w;
            if (mode == 200) {
                w = r;  // pure random
            } else if (mode == 300) {
                w = 0;  // all zero
            } else if (mode >= 0 && mode <= 15) {
                unsigned bit_mask = (1u << mode) | (1u << (mode + 16));
                w = r & ~bit_mask;
            } else if (mode >= 100 && mode <= 115) {
                int b = mode - 100;
                unsigned bit_mask = (1u << b) | (1u << (b + 16));
                w = (r & ~bit_mask) | bit_mask;
            } else if (mode >= 600 && mode <= 615) {
                // Cumulative low-to-high: force bits 0..(mode-600) to 0
                int top_bit = mode - 600;
                unsigned single_half = (1u << (top_bit + 1)) - 1;  // bits 0..top_bit set
                unsigned bit_mask = single_half | (single_half << 16);
                w = r & ~bit_mask;
            } else if (mode >= 700 && mode <= 715) {
                // Cumulative high-to-low: force bits (15-(mode-700))..15 to 0
                int n_bits = (mode - 700) + 1;
                unsigned single_half = ~((1u << (16 - n_bits)) - 1) & 0xFFFFu;  // top n_bits set
                unsigned bit_mask = single_half | (single_half << 16);
                w = r & ~bit_mask;
            } else if (mode == 800) {
                w = r & ~0x007F007Fu;  // mant only (bits 0-6)
            } else if (mode == 801) {
                w = r & ~0x7F807F80u;  // exp only (bits 7-14)
            } else if (mode == 802) {
                w = r & ~0x7FFF7FFFu;  // mant+exp (bits 0-14)
            } else if (mode == 803) {
                w = r & ~0x80008000u;  // sign only (bit 15)
            } else if (mode == 804) {
                w = r & ~0xFF80FF80u;  // sign+exp (bits 7-15)
            } else if (mode == 805) {
                w = r & ~0x807F807Fu;  // sign+mant (bits 0-6, 15)
            } else if (mode >= 900 && mode <= 1155) {
                // Force entire exp field (bits 7-14) to specific value V = mode - 900
                int V = mode - 900;
                if (V > 255) V = 255;
                unsigned exp_pat = (V & 0xFF) << 7;
                unsigned exp_mask = 0x7F80u;
                w = (r & ~(exp_mask | (exp_mask << 16))) | (exp_pat | (exp_pat << 16));
            } else if (mode == 1200) {
                // B = pure +Inf (sign=0, exp=255, mant=0)
                w = 0x7F807F80u;
            } else if (mode == 1201) {
                // B = pure -Inf (sign=1, exp=255, mant=0)
                w = 0xFF80FF80u;
            } else if (mode == 1202) {
                // B = +Inf with random sign (still Inf, just ±)
                w = (r & 0x80008000u) | 0x7F807F80u;
            } else if (mode == 1203) {
                // B = NaN with mant=0x7F (max), sign=0, exp=255
                w = 0x7FFF7FFFu;
            } else if (mode == 1204) {
                // B = NaN with mant=0x40 (mid), sign=0, exp=255
                w = 0x7FC07FC0u;
            } else if (mode == 1205) {
                // B = NaN with random mant, sign=0, exp=255
                w = (r & 0x007F007Fu) | 0x7F807F80u;
                w |= 0x00010001u;
            } else if (mode == 1400) {
                // B = constant +1.0 (s=0 e=127 m=0)
                w = 0x3F803F80u;
            } else if (mode == 1401) {
                // B = constant +2.0 (s=0 e=128 m=0)
                w = 0x40004000u;
            } else if (mode == 1402) {
                // B = constant +6.0 (s=0 e=129 m=0x40)
                w = 0x40C040C0u;
            } else if (mode == 1403) {
                // B = constant +1.5 (s=0 e=127 m=0x40)
                w = 0x3FC03FC0u;
            } else if (mode == 1404) {
                // B = constant -1.0 (s=1 e=127 m=0)
                w = 0xBF80BF80u;
            } else if (mode == 1405) {
                // B = constant +smallest-normal (s=0 e=1 m=0) ~ 2^-126
                w = 0x00800080u;
            } else if (mode == 1406) {
                // B = constant +largest-normal (s=0 e=254 m=0x7F) ~ 2^127
                w = 0x7F7F7F7Fu;
            } else if (mode == 1407) {
                // B = constant subnormal (s=0 e=0 m=0x40) ~ 2^-127 if subnormal
                w = 0x00400040u;
            } else if (mode >= 1500 && mode <= 1510) {
                // K_unique values cycle across N positions, fixed across K (per-MAC temporal constant)
                int K_unique = 1 << (mode - 1500);
                int n_even = (idx % 64) * 2;
                int v_idx_e = n_even % K_unique;
                int v_idx_o = (n_even + 1) % K_unique;
                auto val = [](int v) -> unsigned short {
                    static const unsigned short vals[16] = {
                        0x3F80, 0x4000, 0x40C0, 0x3FC0, 0xBF80, 0x4040, 0x4180, 0x3F00,
                        0x4080, 0xBFC0, 0x4100, 0x3F40, 0x4200, 0xC080, 0x4140, 0x3FE0
                    };
                    return vals[v & 15];
                };
                w = ((unsigned)val(v_idx_o) << 16) | val(v_idx_e);
            } else if (mode >= 1600 && mode <= 1610) {
                // K-vary
                int K_unique_k = 1 << (mode - 1600);
                int k = idx / 64;
                int v_idx = k % K_unique_k;
                auto val = [](int v) -> unsigned short {
                    static const unsigned short vals[16] = {
                        0x3F80, 0x4000, 0x40C0, 0x3FC0, 0xBF80, 0x4040, 0x4180, 0x3F00,
                        0x4080, 0xBFC0, 0x4100, 0x3F40, 0x4200, 0xC080, 0x4140, 0x3FE0
                    };
                    return vals[v & 15];
                };
                unsigned short v = val(v_idx);
                w = ((unsigned)v << 16) | v;
            } else if (mode >= 1900 && mode <= 1910) {
                // KN-vary: each (k, n_pair) gets its own value from table, cycling K_unique
                // Each K-MAC sees DIFFERENT value across K (K-vary), AND different MACs see different values (N-vary)
                int K_unique = 1 << (mode - 1900);
                int k = idx / 64;
                int npair = idx % 64;
                int n_even = npair * 2;
                auto val = [](int v) -> unsigned short {
                    static const unsigned short vals[16] = {
                        0x3F80, 0x4000, 0x40C0, 0x3FC0, 0xBF80, 0x4040, 0x4180, 0x3F00,
                        0x4080, 0xBFC0, 0x4100, 0x3F40, 0x4200, 0xC080, 0x4140, 0x3FE0
                    };
                    return vals[v & 15];
                };
                // FIXED: Use (k + n_even) since k*128 collapsed mod K_unique=16
                unsigned short ve = val((k + n_even) % K_unique);
                unsigned short vo = val((k + n_even + 1) % K_unique);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 2000 && mode <= 2004) {
                // K-vary HIGH-MANTISSA: vals all exp=127 (=1.0), but mantissa pseudo-random.
                // Tests: per-cycle MANTISSA ENTROPY hypothesis (mant popcount drives multiplier energy)
                // K_unique = 1<<(mode-2000) = 1, 2, 4, 8, 16 (cap K=16 for BF16)
                int K_unique = 1 << (mode - 2000);
                int k = idx / 64;
                int v_idx = k % K_unique;
                // High-mantissa table: exp = 127 (0x3F80), mantissa from 0x55, 0x6A pattern (high popcount)
                static const unsigned short himant_vals[16] = {
                    0x3FD5, 0x3FAB, 0x3FF7, 0x3FCD, 0x3FB6, 0x3FE3, 0x3F9A, 0x3FFE,
                    0x3F95, 0x3FE9, 0x3FBC, 0x3FD7, 0x3FA5, 0x3FF2, 0x3FC6, 0x3FBB
                };
                unsigned short v = himant_vals[v_idx & 15];
                w = ((unsigned)v << 16) | v;
            } else if (mode >= 2100 && mode <= 2104) {
                // K-vary HIGH-EXP: vals all mant=0, but exp varied (high popcount in exp)
                // Tests: per-cycle EXP ENTROPY hypothesis
                int K_unique = 1 << (mode - 2100);
                int k = idx / 64;
                int v_idx = k % K_unique;
                // exp varies via high-popcount: 0xAB, 0x55, 0xD5, 0x6A, etc. (avoid 0=subnormal, 0xFF=Inf)
                static const unsigned short hiexp_vals[16] = {
                    0x3D80, 0x4280, 0x3500, 0x4A80, 0x3380, 0x4D80, 0x2D80, 0x5280,
                    0x3700, 0x4880, 0x3D00, 0x4300, 0x3500, 0x4A80, 0x3F00, 0x4080
                };
                unsigned short v = hiexp_vals[v_idx & 15];
                w = ((unsigned)v << 16) | v;
            } else if (mode >= 2200 && mode <= 2204) {
                // K-vary FULL random per K: each K position has truly random 16-bit BF16 value
                // (with random sign+exp+mant - includes some Inf/NaN/subnormal probabilistically)
                // Tests: full-entropy per-K-cycle behavior
                int K_unique = 1 << (mode - 2200);
                int k = idx / 64;
                int v_idx = k % K_unique;
                // Hash-derived random per K position (unique 16-bit pattern per K)
                unsigned hash = 0xC0FFEE13u ^ (v_idx * 0xDEADBEEFu);
                hash = hash * 0x9E3779B1u;
                hash ^= hash >> 16;
                unsigned short v = (unsigned short)(hash & 0xFFFF);
                w = ((unsigned)v << 16) | v;
            } else if (mode >= 2300 && mode <= 2304) {
                // K-vary normal-only random: random sign+mant, exp constrained to 1..254 (avoid sub/Inf/NaN)
                int K_unique = 1 << (mode - 2300);
                int k = idx / 64;
                int v_idx = k % K_unique;
                unsigned hash = 0xC0FFEE13u ^ (v_idx * 0xDEADBEEFu);
                hash = hash * 0x9E3779B1u;
                hash ^= hash >> 16;
                unsigned short v = (unsigned short)(hash & 0xFFFF);
                // Force exp into [1, 254]
                unsigned short e = (v >> 7) & 0xFF;
                if (e == 0) e = 1;
                if (e == 255) e = 254;
                v = (v & 0x807F) | (e << 7);
                w = ((unsigned)v << 16) | v;
            } else if (mode >= 2400 && mode <= 2404) {
                // KN-vary fully random per (k, n_pair): tests if N-vary adds cost when K already varies fully
                // K_unique = 1<<(mode-2400) = 1, 2, 4, 8, 16
                int K_unique = 1 << (mode - 2400);
                int k = idx / 64;
                int npair = idx % 64;
                int n_even = npair * 2;
                // Each (k, n_even) has unique random hash
                auto h = [](int kk, int nn) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (kk * 0xDEADBEEFu) ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                int v_idx_e = (k + n_even) % K_unique;  // K_unique cap (collapsing pattern)
                int v_idx_o = (k + n_even + 1) % K_unique;
                unsigned short ve = h(v_idx_e, 0);
                unsigned short vo = h(v_idx_o, 1);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 2500 && mode <= 2504) {
                // KN-vary INDEPENDENTLY random per K and per N (no collapse): per-(k, n_pair) value depends on BOTH
                // K_unique = 1<<(mode-2500) varies the K RANGE only (cap on K-side variation)
                // N is FULLY random across all 64 N-pairs
                int K_unique = 1 << (mode - 2500);
                int k = idx / 64;
                int npair = idx % 64;
                int n_even = npair * 2;
                int k_idx = k % K_unique;  // K wraps mod K_unique
                int n_idx_e = n_even;       // N never wraps (full 0..127)
                int n_idx_o = n_even + 1;
                auto h = [](int kk, int nn) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (kk * 0xDEADBEEFu) ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                unsigned short ve = h(k_idx, n_idx_e);
                unsigned short vo = h(k_idx, n_idx_o);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 2600 && mode <= 2604) {
                // K-FULL-vary, N-vary with N_unique = 1<<(mode-2600)
                // K is FULLY varying (each K position has distinct random value, K_unique = 16)
                // N variation cap: N_unique = 1, 2, 4, 8, 16
                int N_unique = 1 << (mode - 2600);
                int k = idx / 64;
                int npair = idx % 64;
                int n_even = npair * 2;
                int n_idx_e = n_even % N_unique;
                int n_idx_o = (n_even + 1) % N_unique;
                auto h = [](int kk, int nn) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (kk * 0xDEADBEEFu) ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                unsigned short ve = h(k, n_idx_e);
                unsigned short vo = h(k, n_idx_o);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 2700 && mode <= 2707) {
                // PURE N-vary HIGH-ENTROPY: K constant per row, N varies with cap N_unique
                // N_unique = 1<<(mode-2700) = 1, 2, 4, 8, 16, 32, 64, 128
                int N_unique = 1 << (mode - 2700);
                int npair = idx % 64;
                int n_even = npair * 2;
                int n_idx_e = n_even % N_unique;
                int n_idx_o = (n_even + 1) % N_unique;
                auto h = [](int nn) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                unsigned short ve = h(n_idx_e);
                unsigned short vo = h(n_idx_o);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 2750 && mode <= 2799) {
                // FINE N-vary: N_unique = (mode - 2750), allows non-power-of-2 sweep 17..49
                int N_unique = (mode - 2750);
                if (N_unique < 1) N_unique = 1;
                int npair = idx % 64;
                int n_even = npair * 2;
                int n_idx_e = n_even % N_unique;
                int n_idx_o = (n_even + 1) % N_unique;
                auto h = [](int nn) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                unsigned short ve = h(n_idx_e);
                unsigned short vo = h(n_idx_o);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 3200 && mode <= 3208) {
                // PATTERN COUNT v2 with DIFFERENT HASH (rule out hash-collision artifacts)
                int N_distinct = (mode - 3200);
                if (N_distinct < 1) N_distinct = 1;
                if (N_distinct > 8) N_distinct = 8;
                int npair = idx % 64;
                int sub_tile = npair / 8;
                int pos_in_tile = npair % 8;
                int pattern_id = sub_tile % N_distinct;
                int n_seed_e = pos_in_tile * 2 + pattern_id * 100;
                int n_seed_o = pos_in_tile * 2 + 1 + pattern_id * 100;
                // Murmur-like alternative hash
                auto h2 = [](int nn) -> unsigned short {
                    unsigned x = (unsigned)nn;
                    x ^= x >> 17;
                    x *= 0xED5AD4BBu;
                    x ^= x >> 11;
                    x *= 0xAC4C1B51u;
                    x ^= x >> 15;
                    x *= 0x31848BABu;
                    x ^= x >> 14;
                    return (unsigned short)(x & 0xFFFF);
                };
                unsigned short ve = h2(n_seed_e);
                unsigned short vo = h2(n_seed_o);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 3300 && mode <= 3308) {
                // PATTERN COUNT v3 with FIXED non-hash bytes (use pattern_id directly as repeating byte)
                // Each sub-tile = 32 bytes of value (pattern_id*16 + pos) - deterministic, no hash
                int N_distinct = (mode - 3300);
                if (N_distinct < 1) N_distinct = 1;
                if (N_distinct > 8) N_distinct = 8;
                int npair = idx % 64;
                int sub_tile = npair / 8;
                int pos_in_tile = npair % 8;
                int pattern_id = sub_tile % N_distinct;
                // Each pattern = 32 BF16 values where bytes (pattern_id<<4) | pos
                unsigned short ve = (unsigned short)(((pattern_id & 0xF) << 12) | ((pos_in_tile & 0x7) << 9) | 0x100);  // exp ~127
                unsigned short vo = (unsigned short)(((pattern_id & 0xF) << 12) | ((pos_in_tile & 0x7) << 9) | 0x180);
                // Force exp valid: use bits 7-14 = exp; let me reconstruct as normal value
                // Just force exp=127 (0x3F80 base) and vary mantissa
                ve = 0x3F00 | (pattern_id & 0x7F) | (pos_in_tile << 4);
                vo = 0x3F00 | (((pattern_id+1) & 0x7F)) | (pos_in_tile << 4);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 3100 && mode <= 3108) {
                // PATTERN COUNT test: vary number of DISTINCT sub-tile patterns 1..8
                // N_distinct = mode - 3100; sub_tile uses pattern_id = sub_tile % N_distinct
                int N_distinct = (mode - 3100);
                if (N_distinct < 1) N_distinct = 1;
                if (N_distinct > 8) N_distinct = 8;
                int npair = idx % 64;
                int sub_tile = npair / 8;
                int pos_in_tile = npair % 8;
                int pattern_id = sub_tile % N_distinct;
                int n_seed_e = pos_in_tile * 2 + pattern_id * 100;
                int n_seed_o = pos_in_tile * 2 + 1 + pattern_id * 100;
                auto h = [](int nn) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                unsigned short ve = h(n_seed_e);
                unsigned short vo = h(n_seed_o);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 3020 && mode <= 3027) {
                // SINGLE UNIQUE POSITION: 1 sub-tile unique, 7 shared.
                // mode 3020+P: sub_tile P is unique (using pattern_id P+1), all else use pattern 0.
                // Tests sticky activation: P=0 activates at start (~full cost),
                //                         P=7 activates at end (~minimum cost).
                int P = mode - 3020;
                int npair = idx % 64;
                int sub_tile = npair / 8;
                int pos_in_tile = npair % 8;
                int pattern_id = (sub_tile == P) ? (P + 1) : 0;
                int n_seed_e = pos_in_tile * 2 + pattern_id * 100;
                int n_seed_o = pos_in_tile * 2 + 1 + pattern_id * 100;
                auto h = [](int nn) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                unsigned short ve = h(n_seed_e);
                unsigned short vo = h(n_seed_o);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 3000 && mode <= 3007) {
                // POSITION-INVARIANCE TEST: 4 sub-tiles unique, 4 shared (=pattern_id 0)
                // Different placements:
                // 3000: shared at positions [0,2,4,6], unique at [1,3,5,7] (alternating)
                // 3001: shared at [1,3,5,7], unique at [0,2,4,6] (alternating, opposite phase)
                // 3002: shared at [0,1,2,3], unique at [4,5,6,7] (clustered, equiv to 2904)
                // 3003: shared at [4,5,6,7], unique at [0,1,2,3] (clustered, mirror)
                // 3004: shared at [0,1,4,5], unique at [2,3,6,7] (paired)
                // 3005: shared at [2,3,6,7], unique at [0,1,4,5] (paired, mirror)
                // 3006: shared at [0,1,2,4], unique at [3,5,6,7] (asymmetric A)
                // 3007: shared at [3,4,5,7], unique at [0,1,2,6] (asymmetric B)
                int npair = idx % 64;
                int sub_tile = npair / 8;
                int pos_in_tile = npair % 8;
                int placement_mask;
                if (mode == 3000) placement_mask = 0b10101010;
                else if (mode == 3001) placement_mask = 0b01010101;
                else if (mode == 3002) placement_mask = 0b11110000;
                else if (mode == 3003) placement_mask = 0b00001111;
                else if (mode == 3004) placement_mask = 0b11001100;
                else if (mode == 3005) placement_mask = 0b00110011;
                else if (mode == 3006) placement_mask = 0b11101000;
                else placement_mask = 0b00010111;  // 3007
                int is_unique = (placement_mask >> sub_tile) & 1;
                int pattern_id = is_unique ? sub_tile : 0;
                int n_seed_e = pos_in_tile * 2 + pattern_id * 100;
                int n_seed_o = pos_in_tile * 2 + 1 + pattern_id * 100;
                auto h = [](int nn) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                unsigned short ve = h(n_seed_e);
                unsigned short vo = h(n_seed_o);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 5000 && mode <= 5004) {
                // K-rotating sub-tile pattern test:
                // Within each K row, N=16 unique values (fits cache).
                // Across K rows, the SET of 16 values rotates (k % K_unique selects which set).
                // K_unique = 1<<(mode-5000) = 1, 2, 4, 8, 16
                // Tests: is dedup cache per-K-row (then all free) or per-MMA (then cliff at K_unique > 1)?
                int K_unique = 1 << (mode - 5000);
                int k = idx / 64;
                int npair = idx % 64;
                int n_even = npair * 2;
                int n_idx_e = n_even % 16;     // within each K row, N varies with N_unique=16
                int n_idx_o = (n_even + 1) % 16;
                int k_idx = k % K_unique;
                auto h = [](int kk, int nn) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (kk * 0xDEADBEEFu) ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                unsigned short ve = h(k_idx, n_idx_e);
                unsigned short vo = h(k_idx, n_idx_o);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 5200 && mode <= 5220) {
                // K-row consecutive grouping test: K_unique = (mode - 5200) distinct content
                // patterns arranged in CONSECUTIVE GROUPS of size 16/K_unique each
                // K_unique=1: AAAAAAAAAAAAAAAA - all same (=mode 5000)
                // K_unique=2: AAAAAAAA BBBBBBBB - 2 groups of 8 K rows
                // K_unique=4: AAAA BBBB CCCC DDDD - 4 groups of 4 K rows
                // K_unique=8: AABB CCDD EEFF GGHH - 8 groups of 2 K rows
                // K_unique=16: ABCDEFGH IJKLMNOP - all different (= mode 5004 == K-vary alone)
                int K_unique = (mode - 5200);
                if (K_unique < 1) K_unique = 1;
                if (K_unique > 16) K_unique = 16;
                int k = idx / 64;
                int npair = idx % 64;
                int n_even = npair * 2;
                int n_idx_e = n_even % 16;
                int n_idx_o = (n_even + 1) % 16;
                int group_size = 16 / K_unique;
                if (group_size < 1) group_size = 1;
                int k_group = k / group_size;
                auto h = [](int kk, int nn) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (kk * 0xDEADBEEFu) ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                unsigned short ve = h(k_group, n_idx_e);
                unsigned short vo = h(k_group, n_idx_o);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 5100 && mode <= 5104) {
                // K-rotating WITH WIDER N pattern: each K row has N_unique=32 (above cliff).
                // K rows rotate through K_unique distinct 32-N-pattern sets.
                // If per-MMA cache, free at K_unique=1, cost beyond.
                // If per-K-row, cost at K_unique=1 (since N_unique=32 already over cliff).
                int K_unique = 1 << (mode - 5100);
                int k = idx / 64;
                int npair = idx % 64;
                int n_even = npair * 2;
                int n_idx_e = n_even % 32;
                int n_idx_o = (n_even + 1) % 32;
                int k_idx = k % K_unique;
                auto h = [](int kk, int nn) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (kk * 0xDEADBEEFu) ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                unsigned short ve = h(k_idx, n_idx_e);
                unsigned short vo = h(k_idx, n_idx_o);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 2900 && mode <= 2908) {
                // SUB-TILE DEDUP TEST: 8 sub-tiles of 16 N each.
                // K_break = mode - 2900 sub-tiles use UNIQUE pattern; rest use SHARED pattern 0.
                // K_break=0: all 8 identical → free
                // K_break=7: 1 shared, 7 unique
                // K_break=8: all unique
                // K constant per row.
                int K_break = mode - 2900;
                int npair = idx % 64;
                int sub_tile = npair / 8;       // 0..7
                int pos_in_tile = npair % 8;    // 0..7 (each = 2 N values)
                // Sub-tile uses pattern_id = 0 unless it's in the "broken" set
                int pattern_id;
                if (sub_tile < (8 - K_break)) {
                    pattern_id = 0;  // shared
                } else {
                    pattern_id = sub_tile;  // unique per sub-tile
                }
                int n_seed_e = pos_in_tile * 2 + pattern_id * 100;
                int n_seed_o = pos_in_tile * 2 + 1 + pattern_id * 100;
                auto h = [](int nn) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    return (unsigned short)(hh & 0xFFFF);
                };
                unsigned short ve = h(n_seed_e);
                unsigned short vo = h(n_seed_o);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 2800 && mode <= 2807) {
                // PURE N-vary normal-only HIGH-ENTROPY: same as 2700 but exp clamped to [1,254]
                int N_unique = 1 << (mode - 2800);
                int npair = idx % 64;
                int n_even = npair * 2;
                int n_idx_e = n_even % N_unique;
                int n_idx_o = (n_even + 1) % N_unique;
                auto h_norm = [](int nn) -> unsigned short {
                    unsigned hh = 0xC0FFEE13u ^ (nn * 0x9E3779B1u);
                    hh = hh * 0x85EBCA6Bu;
                    hh ^= hh >> 16;
                    unsigned short v = (unsigned short)(hh & 0xFFFF);
                    unsigned short e = (v >> 7) & 0xFF;
                    if (e == 0) e = 1;
                    if (e == 255) e = 254;
                    return (v & 0x807F) | (e << 7);
                };
                unsigned short ve = h_norm(n_idx_e);
                unsigned short vo = h_norm(n_idx_o);
                w = ((unsigned)vo << 16) | ve;
            } else if (mode >= 4000 && mode <= 4299) {
                // A-test modes: B forced to constant +1.0 to isolate A contribution
                w = 0x3F803F80u;
            } else {
                // For mode >= 400 (A bit forcing), B is random
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

    // VERIFY: print first 4 BF16 values of B if verify flag set
    if (verify && threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < 2; i++) {
            unsigned w = smem_B[i];
            unsigned bf16_lo = w & 0xFFFF;
            unsigned bf16_hi = (w >> 16) & 0xFFFF;
            printf("  B[idx=%d] word=0x%08x lo=0x%04x (s=%d e=%d m=0x%02x) hi=0x%04x (s=%d e=%d m=0x%02x)\n",
                   i, w, bf16_lo,
                   (bf16_lo >> 15) & 1, (bf16_lo >> 7) & 0xFF, bf16_lo & 0x7F,
                   bf16_hi,
                   (bf16_hi >> 15) & 1, (bf16_hi >> 7) & 0xFF, bf16_hi & 0x7F);
        }
    }

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    __syncthreads();
    unsigned tmem_addr = tmem_slot;

    unsigned idesc = (1U << 4) | (1U << 7) | (1U << 10)
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
        printf("BF16 perbit mode=%d iters=%d cy/MMA=%.2f\n",
               mode, iters, (double)(t1-t0)/iters);
    }
}
