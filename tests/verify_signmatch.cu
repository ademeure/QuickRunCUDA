// Verify smem_B layout for the signmatch kernel: print first few values
// for given (sparsity_pct, match_offset) to confirm sign bits are correct.
#define MMA_M 256
#define MMA_N 256
#define MMA_K 96

extern "C" __global__ void init(float* A, float* B, float* C, int u0, int u1, int u2) {}

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int u0, int sparsity_pct, int match_offset) {
    __shared__ unsigned smem_B[3072];
    int smem_size = 3072 / 8;
    int npacks = MMA_N / 8;

    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            int k = idx / npacks;
            int npack = idx % npacks;
            int n_base = npack * 8;
            unsigned r = (idx + blockIdx.x * 1024u + 0xC0FFEE00u) * 0x9E3779B1u;
            r ^= r >> 16; r *= 0x85EBCA6Bu;
            r ^= r >> 13; r *= 0xC2B2AE35u;
            r ^= r >> 16;
            unsigned val = 0;
            for (int p = 0; p < 8; p++) {
                int n = n_base + p;
                unsigned fp4 = (r >> (p * 4)) & 0xF;
                unsigned spr_h = ((unsigned)k * 1664525u + (unsigned)n * 1013904223u + 0xFEEDFACEu);
                spr_h ^= spr_h >> 16; spr_h *= 0xCAFEBABEu;
                spr_h ^= spr_h >> 13;
                bool is_sparse = ((int)(spr_h % 100u)) < sparsity_pct;
                if (is_sparse) {
                    unsigned sign_bit = 0;
                    if (match_offset > 0) {
                        int n_neighbor = (n - match_offset + MMA_N) % MMA_N;
                        int npack_n = n_neighbor / 8;
                        int p_n = n_neighbor % 8;
                        int idx_n = k * npacks + npack_n;
                        unsigned r_n = (idx_n + blockIdx.x * 1024u + 0xC0FFEE00u) * 0x9E3779B1u;
                        r_n ^= r_n >> 16; r_n *= 0x85EBCA6Bu;
                        r_n ^= r_n >> 13; r_n *= 0xC2B2AE35u;
                        r_n ^= r_n >> 16;
                        unsigned fp4_n = (r_n >> (p_n * 4)) & 0xF;
                        unsigned spr_n = ((unsigned)k * 1664525u + (unsigned)n_neighbor * 1013904223u + 0xFEEDFACEu);
                        spr_n ^= spr_n >> 16; spr_n *= 0xCAFEBABEu;
                        spr_n ^= spr_n >> 13;
                        bool n_sparse = ((int)(spr_n % 100u)) < sparsity_pct;
                        if (n_sparse) {
                            sign_bit = 0;
                        } else {
                            sign_bit = (fp4_n >> 3) & 1;
                        }
                    } else if (match_offset < 0) {
                        sign_bit = (fp4 >> 3) & 1;
                    } else {
                        sign_bit = 0;
                    }
                    fp4 = sign_bit << 3;
                }
                val |= (fp4 << (p * 4));
            }
            smem_B[idx] = val;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("sp=%d mo=%d row k=0:\n", sparsity_pct, match_offset);
        printf("  packs[0..7]: ");
        for (int p = 0; p < 8; p++) printf("%08X ", smem_B[p]);
        printf("\n  packs[8..15]: ");
        for (int p = 8; p < 16; p++) printf("%08X ", smem_B[p]);
        printf("\n  packs[16..23]: ");
        for (int p = 16; p < 24; p++) printf("%08X ", smem_B[p]);
        printf("\n  packs[24..31]: ");
        for (int p = 24; p < 32; p++) printf("%08X ", smem_B[p]);
        printf("\n");
        // Element-level: extract 4-bit each
        printf("  elements n=0..63 (FP4 each): ");
        for (int n = 0; n < 64; n++) {
            int npack = n / 8;
            int p = n % 8;
            unsigned fp4 = (smem_B[npack] >> (p * 4)) & 0xF;
            printf("%X", fp4);
            if ((n + 1) % 8 == 0) printf(" ");
        }
        printf("\n");
        printf("  elements n=64..127: ");
        for (int n = 64; n < 128; n++) {
            int npack = n / 8;
            int p = n % 8;
            unsigned fp4 = (smem_B[npack] >> (p * 4)) & 0xF;
            printf("%X", fp4);
            if ((n + 1) % 8 == 0) printf(" ");
        }
        printf("\n");
        // Compare signs at n vs n-64 for sparse elements
        printf("  zero-elt analysis (n: stored_sign / neighbor's_stored_sign at n-64):\n");
        int found = 0;
        for (int n = 64; n < 256 && found < 16; n++) {
            int npack = n / 8;
            int p = n % 8;
            unsigned fp4 = (smem_B[npack] >> (p * 4)) & 0xF;
            unsigned mag = fp4 & 0x7;
            unsigned sign = (fp4 >> 3) & 1;
            if (mag == 0) {
                int n_neighbor = n - 64;
                int npack_n = n_neighbor / 8;
                int p_n = n_neighbor % 8;
                unsigned fp4_n = (smem_B[npack_n] >> (p_n * 4)) & 0xF;
                unsigned mag_n = fp4_n & 0x7;
                unsigned sign_n = (fp4_n >> 3) & 1;
                printf("    n=%3d sign=%d  vs  n=%3d sign=%d (mag_n=%d) %s\n",
                       n, sign, n_neighbor, sign_n, mag_n,
                       (sign == sign_n) ? "MATCH" : "DIFFER");
                found++;
            }
        }
    }
}
