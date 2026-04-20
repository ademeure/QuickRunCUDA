// DRAM-tier popcount test with 64-bit ws_bytes (compile-time WS_BYTES define).
// Supports ws far beyond int32 max.
//
//   -H "#define WS_BYTES 8589934592ULL"   → 8 GiB
//   -H "#define WS_BYTES 34359738368ULL"  → 32 GiB
//
// Args: u0 = iters, u1 = density, u2 = unused (set to 0)
#ifndef WS_BYTES
#define WS_BYTES (8ULL * 1024ULL * 1024ULL * 1024ULL)  // default 8 GiB
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

extern "C" __global__ void init(float* A, float* B, float* C, int u0, int density, int u2) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    unsigned long long n_words = WS_BYTES / 4ULL;
    unsigned* p = (unsigned*)A;
    for (unsigned long long i = idx; i < n_words; i += stride) {
        p[i] = make_word_with_density((unsigned)i, density);
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int iters, int u1, int u2) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long mask = WS_BYTES - 1ULL;
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;
    unsigned acc0=0, acc1=0, acc2=0, acc3=0;
    #pragma unroll 1
    for (int i = 0; i < iters; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned long long off = ((tid * 16ULL + (unsigned long long)(i + j) * 16ULL * total_threads) & mask);
            unsigned long long addr = (unsigned long long)A + off;
            unsigned x0,x1,x2,x3;
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "l"(addr));
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
        }
    }
    ((unsigned*)C)[tid] = acc0 ^ acc1 ^ acc2 ^ acc3;
}
