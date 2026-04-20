// L2 data-dep power: BIT-STRIDE pattern (RUNTIME pattern_mode).
//
// init kernel: arg u0 = pattern_mode (chunks of pattern_mode bits duplicated in pairs)
//   Special: u0 = 0 → all zeros, u0 = 99 → full random (no pairing)
//   u0 = 1, 2, 4, 8, 16, 32, 64, ..., 8192 → bit-stride duplication granularity
//
// Pairs are duplicated with independent random data (deterministic per pair_id).
//
// Main kernel: 16 v4 .cg loads in unrolled inner loop, ~16 TB/s sustained on B300.
// Compile once, run many times with different u0.
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef UNROLL
#define UNROLL 32
#endif

__device__ __forceinline__ unsigned hash_w(unsigned x) {
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    x = x ^ (x >> 16);
    return x;
}

extern "C" __global__ void init(float* A, float* B, float* C, int u0, int pattern_mode, int ws_bytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned* p = (unsigned*)A;
    int n_words = ws_bytes / 4;
    for (int i = idx; i < n_words; i += stride) {
        unsigned v;
        if (pattern_mode == 0) {
            v = 0;
        } else if (pattern_mode == 99) {
            v = hash_w(i);
        } else if (pattern_mode >= 32) {
            // Word-aligned: chunk_words = pattern_mode/32 words per chunk, pair = 2 chunks
            int chunk_words = pattern_mode / 32;
            int pair_words = 2 * chunk_words;
            int pair_id = i / pair_words;
            int word_in_chunk = i % chunk_words;
            v = hash_w(pair_id * chunk_words + word_in_chunk);
        } else if (pattern_mode == 16) {
            unsigned h = hash_w(i);
            unsigned hs = h & 0xFFFF;
            v = hs | (hs << 16);
        } else if (pattern_mode == 8) {
            unsigned hA = hash_w(2*i);
            unsigned hB = hash_w(2*i+1);
            unsigned b0 = hA & 0xFF;
            unsigned b1 = hB & 0xFF;
            v = b0 | (b0 << 8) | (b1 << 16) | (b1 << 24);
        } else if (pattern_mode == 4) {
            unsigned h = hash_w(i);
            v = 0;
            for (int b = 0; b < 4; b++) {
                unsigned nibble = (h >> (b * 4)) & 0xF;
                unsigned byte = nibble | (nibble << 4);
                v |= byte << (b * 8);
            }
        } else if (pattern_mode == 2) {
            unsigned h = hash_w(i);
            v = 0;
            for (int b = 0; b < 4; b++) {
                unsigned c00 = (h >> (b * 4)) & 0x3;
                unsigned c10 = (h >> (b * 4 + 2)) & 0x3;
                unsigned byte = c00 | (c00 << 2) | (c10 << 4) | (c10 << 6);
                v |= byte << (b * 8);
            }
        } else if (pattern_mode == 1) {
            unsigned h = hash_w(i);
            v = 0;
            for (int b = 0; b < 4; b++) {
                unsigned byte = 0;
                for (int q = 0; q < 4; q++) {
                    unsigned bit = (h >> (b * 4 + q)) & 1;
                    byte |= bit << (2 * q);
                    byte |= bit << (2 * q + 1);
                }
                v |= byte << (b * 8);
            }
        } else {
            v = 0;
        }
        p[i] = v;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int iters, int u1, int ws_bytes) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned mask = (unsigned)(ws_bytes - 1);
    unsigned acc0=0, acc1=0, acc2=0, acc3=0;
    #pragma unroll 1
    for (int i = 0; i < iters; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned off = ((tid * 16 + (i + j) * 16 * gridDim.x * blockDim.x) & mask);
            unsigned long long addr = (unsigned long long)A + off;
            unsigned x0,x1,x2,x3;
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "l"(addr));
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
        }
    }
    ((unsigned*)C)[tid] = acc0 ^ acc1 ^ acc2 ^ acc3;
}
