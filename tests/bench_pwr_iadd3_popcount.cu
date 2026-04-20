__device__ __forceinline__ unsigned hash_w(unsigned x) {
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    x = x ^ (x >> 16);
    return x;
}
__device__ __forceinline__ unsigned make_word(unsigned i, int density) {
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
extern "C" __global__ void init(float* A, float* B, float* C, int u0, int u1, int u2) {}
extern "C" __global__ __launch_bounds__(128, 1)
void kernel(float* A, float* B, float* C, int iters, int density, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned a0 = make_word(tid * 8 + 0, density);
    unsigned a1 = make_word(tid * 8 + 1, density);
    unsigned a2 = make_word(tid * 8 + 2, density);
    unsigned a3 = make_word(tid * 8 + 3, density);
    unsigned b0 = make_word(tid * 8 + 4, density);
    unsigned b1 = make_word(tid * 8 + 5, density);
    unsigned b2 = make_word(tid * 8 + 6, density);
    unsigned b3 = make_word(tid * 8 + 7, density);
    unsigned c0 = 0u, c1 = 0u, c2 = 0u, c3 = 0u;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        // IADD3 a, b, c → 3-input add
        asm volatile("iadd3 %0, %1, %2, %3;" : "=r"(c0) : "r"(a0), "r"(b0), "r"(c0));
        asm volatile("iadd3 %0, %1, %2, %3;" : "=r"(c1) : "r"(a1), "r"(b1), "r"(c1));
        asm volatile("iadd3 %0, %1, %2, %3;" : "=r"(c2) : "r"(a2), "r"(b2), "r"(c2));
        asm volatile("iadd3 %0, %1, %2, %3;" : "=r"(c3) : "r"(a3), "r"(b3), "r"(c3));
        asm volatile("iadd3 %0, %1, %2, %3;" : "=r"(c0) : "r"(a3), "r"(b2), "r"(c0));
        asm volatile("iadd3 %0, %1, %2, %3;" : "=r"(c1) : "r"(a2), "r"(b1), "r"(c1));
        asm volatile("iadd3 %0, %1, %2, %3;" : "=r"(c2) : "r"(a1), "r"(b0), "r"(c2));
        asm volatile("iadd3 %0, %1, %2, %3;" : "=r"(c3) : "r"(a0), "r"(b3), "r"(c3));
    }
    if (c0 + c1 + c2 + c3 == 0u) ((unsigned*)C)[tid] = c0;
}
