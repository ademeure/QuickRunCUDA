// V10: Real sum-reduction kernel (32M floats → 1 float)
// Uses SMEM + warp-reductions + SHFL + final global atomic
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

template<int THREADS>
__global__ __launch_bounds__(THREADS, 1)
void reduce(const float* __restrict__ A, float* result, int N) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride load with accumulate
    float sum = 0.0f;
    for (int i = gtid; i < N; i += stride) {
        sum += A[i];
    }

    // Warp reduce via SHFL.BFLY
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, off);
    }

    // Cross-warp reduce via SMEM
    __shared__ float ssum[THREADS / 32];
    if ((tid & 31) == 0) ssum[tid >> 5] = sum;
    __syncthreads();

    if (tid < (THREADS / 32)) {
        sum = ssum[tid];
        #pragma unroll
        for (int off = (THREADS / 32) / 2; off > 0; off >>= 1) {
            sum += __shfl_xor_sync((1u << (THREADS / 32)) - 1, sum, off);
        }
        if (tid == 0) atomicAdd(result, sum);
    }
}

int main() {
    cudaSetDevice(0);

    int N = 32 * 1024 * 1024;
    float* d_A;
    float* d_result;
    CK(cudaMalloc(&d_A, N * sizeof(float)));
    CK(cudaMalloc(&d_result, sizeof(float)));

    // Init
    float* h_A = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_A[i] = (float)(i & 0xFF) * 0.001f;
    CK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = 148 * 8;   // ~full occupancy
    int threads = 256;

    // Warmup
    for (int i = 0; i < 3; i++) {
        cudaMemsetAsync(d_result, 0, 4);
        reduce<256><<<blocks, threads>>>(d_A, d_result, N);
    }
    cudaDeviceSynchronize();

    // Measure
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 10; i++) {
        cudaMemsetAsync(d_result, 0, 4);
        reduce<256><<<blocks, threads>>>(d_A, d_result, N);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    ms /= 10;

    float result;
    cudaMemcpy(&result, d_result, 4, cudaMemcpyDeviceToHost);

    // Verify
    float expected = 0;
    for (int i = 0; i < N; i++) expected += h_A[i];

    double bytes = (double)N * sizeof(float);
    double gbs = bytes / 1e9 / (ms / 1000.0);

    printf("=== Sum reduction (32M floats = 128 MB) ===\n");
    printf("  Time: %.3f ms\n", ms);
    printf("  BW: %.2f GB/s (%.1f%% of 7200 peak)\n", gbs, gbs * 100.0 / 7200);
    printf("  Blocks: %d × %d = %d threads\n", blocks, threads, blocks * threads);
    printf("  Result: %.6f (expected %.6f, diff %.2e)\n",
           result, expected, fabsf(result - expected));

    cudaFree(d_A); cudaFree(d_result);
    free(h_A);
    return 0;
}
