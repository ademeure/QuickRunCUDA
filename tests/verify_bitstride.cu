// Verify bitstride init produces correct duplication pattern.
// Reuses init from bench_l2_bitstride.cu logic, then prints first few words/bytes.
#ifndef PATTERN_MODE
#define PATTERN_MODE 8
#endif

__device__ __forceinline__ unsigned hash_w(unsigned x) {
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    x = x ^ (x >> 16);
    return x;
}

extern "C" __global__ void init(float* A, float* B, float* C, int u0, int u1, int n_words) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned* p = (unsigned*)A;
    for (int i = idx; i < n_words; i += stride) {
        unsigned v;
#if PATTERN_MODE == 0
        v = 0;
#elif PATTERN_MODE == 99
        v = hash_w(i);
#elif PATTERN_MODE >= 32
        constexpr int chunk_words = PATTERN_MODE / 32;
        constexpr int pair_words = 2 * chunk_words;
        int pair_id = i / pair_words;
        int word_in_chunk = i % chunk_words;
        v = hash_w(pair_id * chunk_words + word_in_chunk);
#elif PATTERN_MODE == 16
        unsigned h = hash_w(i);
        unsigned short hs = h & 0xFFFF;
        v = ((unsigned)hs) | (((unsigned)hs) << 16);
#elif PATTERN_MODE == 8
        unsigned hA = hash_w(2*i);
        unsigned hB = hash_w(2*i+1);
        unsigned char b0 = hA & 0xFF;
        unsigned char b1 = hB & 0xFF;
        v = (unsigned)b0 | ((unsigned)b0 << 8) | ((unsigned)b1 << 16) | ((unsigned)b1 << 24);
#elif PATTERN_MODE == 4
        unsigned h = hash_w(i);
        v = 0;
        for (int b = 0; b < 4; b++) {
            unsigned char nibble = (h >> (b * 4)) & 0xF;
            unsigned char byte = nibble | (nibble << 4);
            v |= ((unsigned)byte) << (b * 8);
        }
#elif PATTERN_MODE == 2
        unsigned h = hash_w(i);
        v = 0;
        for (int b = 0; b < 4; b++) {
            unsigned c00 = (h >> (b * 4)) & 0x3;
            unsigned c10 = (h >> (b * 4 + 2)) & 0x3;
            unsigned char byte = c00 | (c00 << 2) | (c10 << 4) | (c10 << 6);
            v |= ((unsigned)byte) << (b * 8);
        }
#elif PATTERN_MODE == 1
        unsigned h = hash_w(i);
        v = 0;
        for (int b = 0; b < 4; b++) {
            unsigned char byte = 0;
            for (int p = 0; p < 4; p++) {
                int bit = (h >> (b * 4 + p)) & 1;
                byte |= bit << (2 * p);
                byte |= bit << (2 * p + 1);
            }
            v |= ((unsigned)byte) << (b * 8);
        }
#else
        v = 0;
#endif
        p[i] = v;
    }
}

extern "C" __global__ void kernel(float* A, float* B, float* C, int u0, int u1, int u2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned* p = (unsigned*)A;
        printf("PATTERN_MODE=%d\n", PATTERN_MODE);
        printf("words[0..15]:");
        for (int i = 0; i < 16; i++) printf(" %08X", p[i]);
        printf("\n");
        unsigned char* b = (unsigned char*)A;
        printf("bytes[0..31]:");
        for (int i = 0; i < 32; i++) printf(" %02X", b[i]);
        printf("\n");
        printf("bytes[1024..1055]:");
        for (int i = 1024; i < 1056; i++) printf(" %02X", b[i]);
        printf("\n");
    }
}
