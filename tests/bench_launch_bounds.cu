// __launch_bounds__ impact on register count.
// Different (max_threads_per_block, min_blocks_per_sm) tuples affect register limit.
// Check what nvcc emits for each.

#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
#elif MODE == 1
extern "C" __global__ __launch_bounds__(32, 1) void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
#elif MODE == 2
extern "C" __global__ __launch_bounds__(256, 4) void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
#elif MODE == 3
extern "C" __global__ __launch_bounds__(1024, 1) void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
#elif MODE == 4
extern "C" __global__ __launch_bounds__(1024, 2) void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
#endif
    // Heavy register pressure
    unsigned int v[16];
    #pragma unroll
    for (int k = 0; k < 16; k++) v[k] = (unsigned)(threadIdx.x + k + (unsigned)u2);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < 16; k++) v[k] = v[k] * 31u + (unsigned)i;
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < 16; k++) acc ^= v[k];
    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
