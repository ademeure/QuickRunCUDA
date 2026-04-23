// §13 verification: does per-thread predication change pipe throughput?
// Run FFMA chain with: full warp, half warp, 1 lane active. Measure ncu pipe_fma rate.
//
// -H "#define ACTIVE_MASK 0xFFFFFFFF"  (or 0x0000FFFF, 0x00000001)

#ifndef N_CHAINS
#define N_CHAINS 16
#endif
#ifndef ITERS
#define ITERS 4096
#endif
#ifndef ACTIVE_MASK
#define ACTIVE_MASK 0xFFFFFFFF
#endif

extern "C" __global__ __launch_bounds__(128, 4)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    int lane = threadIdx.x & 31;

    // Predicate: lane is in active set?
    int active = ((1u << lane) & (unsigned)ACTIVE_MASK) != 0;

    float v[N_CHAINS];
    #pragma unroll
    for (int i = 0; i < N_CHAINS; i++)
        v[i] = (float)(lane + i + seed) * 0.001f + 1.0f;

    float k = 1.0001f;
    float b = 1.0f;

    if (active) {
        #pragma unroll 1
        for (int i = 0; i < ITERS; i++) {
            #pragma unroll
            for (int j = 0; j < N_CHAINS; j++) {
                v[j] = v[j] * k + b;
            }
        }
    }

    float sum = 0;
    #pragma unroll
    for (int i = 0; i < N_CHAINS; i++) sum += v[i];
    if (sum == 0xDEADBEEF) C[lane] = sum;
}
