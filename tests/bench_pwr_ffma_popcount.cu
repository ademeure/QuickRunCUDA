// FFMA power vs operand popcount.
// All threads compute lots of FFMAs with operands of controlled popcount.
// Operands stored in registers, computed once at init via density.
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
    // 8 distinct operands per thread, popcount d, varying bit positions
    float a0 = __uint_as_float(make_word(tid * 8 + 0, density));
    float a1 = __uint_as_float(make_word(tid * 8 + 1, density));
    float a2 = __uint_as_float(make_word(tid * 8 + 2, density));
    float a3 = __uint_as_float(make_word(tid * 8 + 3, density));
    float b0 = __uint_as_float(make_word(tid * 8 + 4, density));
    float b1 = __uint_as_float(make_word(tid * 8 + 5, density));
    float b2 = __uint_as_float(make_word(tid * 8 + 6, density));
    float b3 = __uint_as_float(make_word(tid * 8 + 7, density));
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        c0 = fmaf(a0, b0, c0); c1 = fmaf(a1, b1, c1);
        c2 = fmaf(a2, b2, c2); c3 = fmaf(a3, b3, c3);
        c0 = fmaf(a3, b2, c0); c1 = fmaf(a2, b1, c1);
        c2 = fmaf(a1, b0, c2); c3 = fmaf(a0, b3, c3);
    }
    if (c0 + c1 + c2 + c3 == 0.0f) ((float*)C)[tid] = c0;
}
