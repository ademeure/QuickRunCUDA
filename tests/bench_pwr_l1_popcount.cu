// L1 read power vs popcount: each block reads its own small slice repeatedly (L1-resident).
// Uses .ca cache hint (default L1+L2). Per-SM working set fits in L1 (228 KB).
//
// Args:
//   u0 = iters (outer loop count)
//   u1 = density (bits/dword)
//   u2 = per_block_bytes (size of slice each block reads)
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

extern "C" __global__ void init(float* A, float* B, float* C, int u0, int density, int per_block_bytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int per_block_words = per_block_bytes / 4;
    int total_words = per_block_words * gridDim.x;
    unsigned* p = (unsigned*)A;
    for (int i = idx; i < total_words; i += stride) {
        p[i] = make_word_with_density((unsigned)i, density);
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int iters, int u1, int per_block_bytes) {
    unsigned tid = threadIdx.x;
    unsigned bid = blockIdx.x;
    unsigned mask = (unsigned)(per_block_bytes - 1);   // power-of-2
    // Each block's base = bid * per_block_bytes
    unsigned long long base_addr = (unsigned long long)A + (unsigned long long)bid * per_block_bytes;
    unsigned acc0=0, acc1=0, acc2=0, acc3=0;
    #pragma unroll 1
    for (int i = 0; i < iters; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned off = ((tid * 16 + (i + j) * 16 * blockDim.x) & mask);
            unsigned long long addr = base_addr + off;
            unsigned x0,x1,x2,x3;
            // .ca = default (L1 + L2), no .cg → L1 will cache
            asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "l"(addr));
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
        }
    }
    ((unsigned*)C)[blockIdx.x * blockDim.x + tid] = acc0 ^ acc1 ^ acc2 ^ acc3;
}
