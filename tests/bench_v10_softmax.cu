// V10: Softmax kernel — end-to-end rigor measurement
// 1 block per row, each block reduces max + sum-of-exp, then normalizes
// Row length = N. Common pattern in attention/classifier layers.
#ifndef N
#define N 4096
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* row_in = A + row * N;
    float* row_out = C + row * N;

    // Phase 1: find max
    float tmax = -3.4e38f;  // ~-FLT_MAX (NVRTC has no INFINITY)
    for (int i = tid; i < N; i += stride) {
        float x = row_in[i];
        tmax = fmaxf(tmax, x);
    }
    // Warp reduce max
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        tmax = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFF, tmax, off));
    }
    __shared__ float smax[8];
    if ((tid & 31) == 0) smax[tid >> 5] = tmax;
    __syncthreads();
    tmax = smax[tid & 7];
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        tmax = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFF, tmax, off));
    }
    // Now all threads have max

    // Phase 2: sum of exp(x - max)
    float tsum = 0.0f;
    for (int i = tid; i < N; i += stride) {
        float x = row_in[i];
        tsum += __expf(x - tmax);
    }
    // Warp reduce sum
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        tsum += __shfl_xor_sync(0xFFFFFFFF, tsum, off);
    }
    __shared__ float ssum[8];
    if ((tid & 31) == 0) ssum[tid >> 5] = tsum;
    __syncthreads();
    tsum = ssum[tid & 7];
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        tsum += __shfl_xor_sync(0xFFFFFFFF, tsum, off);
    }
    float inv_sum = 1.0f / tsum;

    // Phase 3: write normalized
    for (int i = tid; i < N; i += stride) {
        float x = row_in[i];
        row_out[i] = __expf(x - tmax) * inv_sum;
    }
}

// Init kernel to fill A with random data
extern "C" __global__ void init(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    for (int i = tid; i < ITERS * N; i += total) {
        A[i] = ((float)((i * 2654435761u) & 0xFFFF) / 65536.0f) - 0.5f;
    }
}
