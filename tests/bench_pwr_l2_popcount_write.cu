// L2 WRITE power vs popcount (with value table for speed).
// Precompute 32 values of popcount d (different bit positions), then in the
// hot loop just rotate through them — keeps inter-cycle toggle activity
// at the bus while making each store itself cheap.
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

extern "C" __global__ void init(float* A, float* B, float* C, int u0, int u1, int ws_bytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n_words = ws_bytes / 4;
    unsigned* p = (unsigned*)A;
    for (int i = idx; i < n_words; i += stride) p[i] = 0u;
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int iters, int density, int ws_bytes) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned mask = (unsigned)(ws_bytes - 1);

    // Precompute 32 distinct popcount-d values per thread (cheap shuffle done once).
    unsigned v[32];
    #pragma unroll
    for (int k = 0; k < 32; k++) v[k] = make_word_with_density(tid * 32u + (unsigned)k, density);

    #pragma unroll 1
    for (int i = 0; i < iters; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned off = ((tid * 16 + (i + j) * 16 * gridDim.x * blockDim.x) & mask);
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
