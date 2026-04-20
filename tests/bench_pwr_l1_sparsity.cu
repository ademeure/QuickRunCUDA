// L1-resident sparsity test (per-block 64 KB, .ca cache hint).
// Random base; X% of "elements" of granularity G replaced by constant V.
//   u0 = iters; u1 = (sparsity_pct<<16)|(gran<<8)|value_kind; u2 = per_block_bytes
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

extern "C" __global__ void init(float* A, float* B, float* C, int u0, int packed, int per_block_bytes) {
    int sparsity_pct = (packed >> 16) & 0xFFFF;
    int gran         = (packed >> 8) & 0xFF;
    int value_kind   = packed & 0xFF;
    if (gran <= 0) gran = 1;
    unsigned char rv = (value_kind == 0) ? 0x00 : (value_kind == 1) ? 0xFF : 0x55;
    unsigned wv = (unsigned)rv | ((unsigned)rv << 8) | ((unsigned)rv << 16) | ((unsigned)rv << 24);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int per_block_words = per_block_bytes / 4;
    int total_words = per_block_words * gridDim.x;
    unsigned* p = (unsigned*)A;

    for (int i = idx; i < total_words; i += stride) {
        // Per-block local index
        int local_word = i % per_block_words;
        int block_id   = i / per_block_words;
        unsigned out;
        if (gran >= 4) {
            int local_byte = local_word * 4;
            int g = local_byte / gran;            // group index within block
            int global_g = block_id * (per_block_bytes / gran) + g;
            unsigned h = hash_w((unsigned)global_g + 0xC0FFEE00u);
            if ((int)(h % 100u) < sparsity_pct) out = wv;
            else out = hash_w((unsigned)i);
        } else {
            unsigned random_w = hash_w((unsigned)i);
            out = 0;
            int local_byte = local_word * 4;
            int global_byte_base = block_id * per_block_bytes + local_byte;
            for (int b = 0; b < 4; b++) {
                unsigned char rb = (unsigned char)((random_w >> (b * 8)) & 0xFF);
                unsigned hh = hash_w((unsigned)(global_byte_base + b) + 0xC0FFEE00u);
                if ((int)(hh % 100u) < sparsity_pct) rb = rv;
                out |= ((unsigned)rb) << (b * 8);
            }
        }
        p[i] = out;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int iters, int u1, int per_block_bytes) {
    unsigned tid = threadIdx.x;
    unsigned bid = blockIdx.x;
    unsigned mask = (unsigned)(per_block_bytes - 1);
    unsigned long long base_addr = (unsigned long long)A + (unsigned long long)bid * per_block_bytes;
    unsigned acc0=0, acc1=0, acc2=0, acc3=0;
    #pragma unroll 1
    for (int i = 0; i < iters; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned off = ((tid * 16 + (i + j) * 16 * blockDim.x) & mask);
            unsigned long long addr = base_addr + off;
            unsigned x0,x1,x2,x3;
            asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "l"(addr));
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
        }
    }
    ((unsigned*)C)[blockIdx.x * blockDim.x + tid] = acc0 ^ acc1 ^ acc2 ^ acc3;
}
