// L2/DRAM data-dep power: SPARSITY sweep.
// Random base data; X% of "elements" of granularity G are replaced by a constant value V.
//   gran (bytes): 1 (byte), 4 (dword), 32 (32-byte chunk), 128 (128-byte chunk)
//   sparsity_pct: 0..100
//   value_kind: 0=zeros (0x00), 1=all-ones (0xFF), 2=alt (0x55)
//
// Arg layout (so main and init can share):
//   u0 = iters       (main reads this; init ignores)
//   u1 = (sparsity_pct<<16) | (gran<<8) | value_kind
//   u2 = ws_bytes
//
// Race-free: each thread emits the FULL value of its assigned word in one shot.

__device__ __forceinline__ unsigned hash_w(unsigned x) {
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    x = x ^ (x >> 16);
    return x;
}

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef UNROLL
#define UNROLL 32
#endif

extern "C" __global__ void init(float* A, float* B, float* C, int u0, int packed, int ws_bytes) {
    int sparsity_pct = (packed >> 16) & 0xFFFF;
    int gran         = (packed >> 8) & 0xFF;          // 1, 4, 32, 128
    int value_kind   = packed & 0xFF;                  // 0, 1, 2
    if (gran <= 0) gran = 1;
    unsigned char rv;
    if (value_kind == 0) rv = 0x00;
    else if (value_kind == 1) rv = 0xFF;
    else rv = 0x55;
    unsigned wv = (unsigned)rv | ((unsigned)rv << 8) | ((unsigned)rv << 16) | ((unsigned)rv << 24);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n_words = ws_bytes / 4;
    unsigned* p = (unsigned*)A;

    for (int i = idx; i < n_words; i += stride) {
        unsigned out;
        if (gran >= 4) {
            // Group g = (byte address) / gran. Same decision for all words in group.
            int g = (i * 4) / gran;
            unsigned h = hash_w((unsigned)g + 0xC0FFEE00u);
            if ((int)(h % 100u) < sparsity_pct) {
                out = wv;
            } else {
                out = hash_w((unsigned)i);
            }
        } else {
            // gran == 1: each of the 4 bytes in this word independently sparse.
            unsigned random_w = hash_w((unsigned)i);
            out = 0;
            int base_byte = i * 4;
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
