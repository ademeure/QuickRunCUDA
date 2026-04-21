// V10: LDG.E width comparison — 32 vs 64 vs 128 bit
// Key question: does wider load amortize address generation?
// All variants read SAME total bytes from HBM; compare time + ncu sectors.
#ifndef WIDTH
#define WIDTH 128  // 32, 64, 128 bits
#endif
#ifndef K_INNER
#define K_INNER 64
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int THREADS = gridDim.x * blockDim.x;
    long long T = THREADS;

    // Always read same total bytes across variants
    // WIDTH determines bytes-per-thread-load
    // If WIDTH=128 (16 B): K_INNER loads per iter = 16B × K_INNER
    // If WIDTH=64  (8 B):  need 2× K_INNER to match 128-bit total
    // If WIDTH=32  (4 B):  need 4× K_INNER to match

#if WIDTH == 128
    #define MULT 1
    #define STRIDE 16
    typedef float4 vtype;
#elif WIDTH == 64
    #define MULT 2
    #define STRIDE 8
    typedef float2 vtype;
#elif WIDTH == 32
    #define MULT 4
    #define STRIDE 4
    typedef float vtype;
#endif

    vtype* Av = (vtype*)A;
    vtype acc{};

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 8
        for (int k = 0; k < K_INNER * MULT; k++) {
            long long idx = (long long)i * T * K_INNER * MULT + (long long)k * T + (long long)gtid;
            vtype v = Av[idx];
#if WIDTH == 128
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
#elif WIDTH == 64
            acc.x += v.x; acc.y += v.y;
#elif WIDTH == 32
            acc += v;
#endif
        }
    }

#if WIDTH == 128
    if (acc.x == 1.234567e-30f) ((float*)C)[gtid] = acc.x;
#elif WIDTH == 64
    if (acc.x == 1.234567e-30f) ((float*)C)[gtid] = acc.x;
#elif WIDTH == 32
    if (acc == 1.234567e-30f) ((float*)C)[gtid] = acc;
#endif
}
