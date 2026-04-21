// Verify init kernel actually sets the patterns it claims to
#ifndef PATTERN_MODE
#define PATTERN_MODE 0
#endif

extern "C" __global__ void init(float* A, float* B, float* C, int u0, int u1, int n_words) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned* p = (unsigned*)A;
    for (int i = idx; i < n_words; i += stride) {
        unsigned v;
        switch (PATTERN_MODE) {
            case 0: v = 0x12121212u; break;
            case 6: v = 0; break;
            case 7: v = 0xFFFFFFFFu; break;
            case 10: {
                unsigned x = i;
                x = (x ^ (x >> 16)) * 0x7feb352du;
                x = (x ^ (x >> 15)) * 0x846ca68bu;
                x = x ^ (x >> 16);
                v = x;
                break;
            }
            default: v = 0;
        }
        p[i] = v;
    }
}

extern "C" __global__ void kernel(float* A, float* B, float* C, int u0, int u1, int u2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned* p = (unsigned*)A;
        printf("PATTERN_MODE=%d sample: A[0]=%08X A[1]=%08X A[100]=%08X A[10000]=%08X\n",
               PATTERN_MODE, p[0], p[1], p[100], p[10000]);
    }
}
