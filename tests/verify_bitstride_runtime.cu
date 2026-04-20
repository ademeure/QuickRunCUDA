// Verify runtime-arg bitstride init produces expected duplication pattern.
// Reads u1 = pattern_mode, u2 = ws_bytes.
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
        if (pattern_mode == 0) v = 0;
        else if (pattern_mode == 99) v = hash_w(i);
        else if (pattern_mode >= 32) {
            int chunk_words = pattern_mode / 32;
            int pair_words = 2 * chunk_words;
            int pair_id = i / pair_words;
            int word_in_chunk = i % chunk_words;
            v = hash_w(pair_id * chunk_words + word_in_chunk);
        }
        else if (pattern_mode == 16) { unsigned h = hash_w(i); unsigned hs = h & 0xFFFF; v = hs | (hs<<16); }
        else if (pattern_mode == 8) {
            unsigned hA = hash_w(2*i); unsigned hB = hash_w(2*i+1);
            unsigned b0 = hA & 0xFF; unsigned b1 = hB & 0xFF;
            v = b0 | (b0<<8) | (b1<<16) | (b1<<24);
        }
        else if (pattern_mode == 4) {
            unsigned h = hash_w(i); v = 0;
            for (int b=0;b<4;b++) { unsigned n = (h>>(b*4))&0xF; unsigned by = n|(n<<4); v |= by<<(b*8); }
        }
        else if (pattern_mode == 2) {
            unsigned h = hash_w(i); v = 0;
            for (int b=0;b<4;b++) {
                unsigned c00 = (h>>(b*4))&0x3; unsigned c10 = (h>>(b*4+2))&0x3;
                unsigned by = c00 | (c00<<2) | (c10<<4) | (c10<<6);
                v |= by<<(b*8);
            }
        }
        else if (pattern_mode == 1) {
            unsigned h = hash_w(i); v = 0;
            for (int b=0;b<4;b++) {
                unsigned by = 0;
                for (int q=0;q<4;q++) {
                    unsigned bit = (h>>(b*4+q))&1;
                    by |= bit<<(2*q); by |= bit<<(2*q+1);
                }
                v |= by<<(b*8);
            }
        }
        else v = 0;
        p[i] = v;
    }
}

extern "C" __global__ void kernel(float* A, float* B, float* C, int u0, int pattern_mode, int u2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned* p = (unsigned*)A;
        printf("p=%d w[0..7]: %08X %08X %08X %08X %08X %08X %08X %08X\n",
               pattern_mode, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    }
}
