// DRAM WRITE power vs popcount with 64-bit ws (compile-time WS_BYTES).
#ifndef WS_BYTES
#define WS_BYTES (8ULL * 1024ULL * 1024ULL * 1024ULL)
#endif
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
__device__ __forceinline__ unsigned make_word_with_density(unsigned i, int density) {
    if (density <= 0) return 0u;
    if (density >= 32) return 0xFFFFFFFFu;
    unsigned char pos[32];
    for (int k = 0; k < 32; k++) pos[k] = (unsigned char)k;
    unsigned r = hash_w(i);
    for (int k = 31; k > 0; k--) {
        r = r * 1103515245u + 12345u;
        unsigned j = r % (k + 1);
        unsigned char t = pos[k]; pos[k] = pos[j]; pos[j] = t;
    }
    unsigned w = 0;
    for (int k = 0; k < density; k++) w |= (1u << pos[k]);
    return w;
}

extern "C" __global__ void init(float* A, float* B, float* C, int u0, int u1, int u2) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    unsigned long long n_words = WS_BYTES / 4ULL;
    unsigned* p = (unsigned*)A;
    for (unsigned long long i = idx; i < n_words; i += stride) p[i] = 0u;
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int iters, int density, int u2) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long mask = WS_BYTES - 1ULL;
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;

    unsigned v[32];
    #pragma unroll
    for (int k = 0; k < 32; k++) v[k] = make_word_with_density((unsigned)tid * 32u + (unsigned)k, density);

    #pragma unroll 1
    for (int i = 0; i < iters; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned long long off = ((tid * 16ULL + (unsigned long long)(i + j) * 16ULL * total_threads) & mask);
            unsigned long long addr = (unsigned long long)A + off;
            int idx0 = (i + j) & 31;
            int idx1 = (i + j + 8) & 31;
            int idx2 = (i + j + 16) & 31;
            int idx3 = (i + j + 24) & 31;
            asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};"
                : : "l"(addr),
                    "r"(v[idx0]), "r"(v[idx1]), "r"(v[idx2]), "r"(v[idx3]));
        }
    }
}
