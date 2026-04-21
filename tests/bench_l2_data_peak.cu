// L2/DRAM data-dep power test, PEAK BW version (16 v4 chain)
// Adapted from bench_l2_peak structure
#ifndef PATTERN_MODE
#define PATTERN_MODE 0
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef UNROLL
#define UNROLL 16
#endif

extern "C" __global__ void init(float* A, float* B, float* C, int u0, int u1, int n_words) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned* p = (unsigned*)A;
    for (int i = idx; i < n_words; i += stride) {
        unsigned v;
        switch (PATTERN_MODE) {
            case 0: v = 0x12121212u; break;
            case 1: v = 0xFF00FF00u; break;
            case 2: { unsigned x = i * 0x9E3779B1u + 0xCAFEBABEu; x ^= x >> 16; x *= 0xCAFEBABEu; v = x; break; }
            case 3: { int wi = i % 8; unsigned x = wi * 0x9E3779B1u + 0xCAFEBABEu; x ^= x >> 16; x *= 0xCAFEBABEu; v = x; break; }
            case 4: { int wi = i % 32; unsigned x = wi * 0x9E3779B1u + 0xCAFEBABEu; x ^= x >> 16; x *= 0xCAFEBABEu; v = x; break; }
            case 5: { int wi = i % 256; unsigned x = wi * 0x9E3779B1u + 0xCAFEBABEu; x ^= x >> 16; x *= 0xCAFEBABEu; v = x; break; }
            case 6: v = 0; break;
            case 7: v = 0xFFFFFFFFu; break;
            case 8: v = 0xCAFEBABEu; break;
            case 9: v = (i & 1) ? 0xFFFFFFFFu : 0u; break;
            case 10: { unsigned x = i; x = (x ^ (x >> 16)) * 0x7feb352du; x = (x ^ (x >> 15)) * 0x846ca68bu; x = x ^ (x >> 16); v = x; break; }
            default: v = 0;
        }
        p[i] = v;
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int iters, int u1, int ws_bytes) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned mask = (unsigned)(ws_bytes - 1);  // ws_bytes must be power-of-2
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
