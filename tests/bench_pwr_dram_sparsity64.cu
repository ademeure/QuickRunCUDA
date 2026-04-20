// DRAM sparsity test with compile-time WS_BYTES (64-bit addressing).
//   u0 = iters; u1 = (sparsity_pct<<16)|(gran<<8)|value_kind; u2 = unused
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

extern "C" __global__ void init(float* A, float* B, float* C, int u0, int packed, int u2) {
    int sparsity_pct = (packed >> 16) & 0xFFFF;
    int gran         = (packed >> 8) & 0xFF;
    int value_kind   = packed & 0xFF;
    if (gran <= 0) gran = 1;
    unsigned char rv = (value_kind == 0) ? 0x00 : (value_kind == 1) ? 0xFF : 0x55;
    unsigned wv = (unsigned)rv | ((unsigned)rv << 8) | ((unsigned)rv << 16) | ((unsigned)rv << 24);

    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    unsigned long long n_words = WS_BYTES / 4ULL;
    unsigned* p = (unsigned*)A;

    for (unsigned long long i = idx; i < n_words; i += stride) {
        unsigned out;
        if (gran >= 4) {
            unsigned long long g = (i * 4ULL) / (unsigned long long)gran;
            unsigned h = hash_w((unsigned)g + 0xC0FFEE00u);
            if ((int)(h % 100u) < sparsity_pct) out = wv;
            else out = hash_w((unsigned)i);
        } else {
            unsigned random_w = hash_w((unsigned)i);
            out = 0;
            unsigned long long base_byte = i * 4ULL;
            for (int b = 0; b < 4; b++) {
                unsigned char rb = (unsigned char)((random_w >> (b * 8)) & 0xFF);
                unsigned hh = hash_w((unsigned)(base_byte + b) + 0xC0FFEE00u);
                if ((int)(hh % 100u) < sparsity_pct) rb = rv;
                out |= ((unsigned)rb) << (b * 8);
            }
        }
        p[i] = out;
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
