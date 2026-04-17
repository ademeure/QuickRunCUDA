// Deeper test: where does the data actually live?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void write_known(float *data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) data[i] = (float)i;
}

extern "C" __global__ void read_check(float *data, int N, int *errors) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int err = 0;
    for (int i = tid; i < N; i += stride) {
        if (data[i] != (float)i) err++;
    }
    if (err > 0) atomicAdd(errors, err);
}

int main() {
    cudaSetDevice(0);

    size_t bytes = 64 * 1024 * 1024;
    int N = bytes / sizeof(float);

    float *p = (float*)aligned_alloc(4096, bytes);
    memset(p, 0, bytes);  // commit pages

    int *d_errors; cudaMalloc(&d_errors, sizeof(int));

    // Warm-up: write from GPU
    write_known<<<148, 256>>>(p, N);
    cudaDeviceSynchronize();

    // Read it back from CPU - if data is HBM-only, CPU sees stale (zeros)
    int cpu_errors = 0;
    for (int i = 0; i < std::min(N, 1000); i++) {
        if (p[i] != (float)i) cpu_errors++;
    }
    printf("CPU read after GPU write: %d/1000 errors (0=GPU writes visible to CPU)\n", cpu_errors);
    printf("  p[0]=%f, p[100]=%f, p[999]=%f\n", p[0], p[100], p[999]);

    // Now GPU read - is the data visible to GPU?
    cudaMemset(d_errors, 0, sizeof(int));
    read_check<<<148, 256>>>(p, N, d_errors);
    cudaDeviceSynchronize();
    int err; cudaMemcpy(&err, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
    printf("GPU read of GPU-written data: %d errors\n", err);

    // CPU writes new value
    for (int i = 0; i < N; i++) p[i] = -1.0f;
    
    // GPU read - is CPU write visible?
    cudaMemset(d_errors, 0, sizeof(int));
    read_check<<<148, 256>>>(p, N, d_errors);
    cudaDeviceSynchronize();
    cudaMemcpy(&err, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
    printf("GPU read after CPU write -1.0: %d errors (high=GPU sees old data, 0=coherent)\n", err);

    // Time the next GPU access
    auto t0 = std::chrono::high_resolution_clock::now();
    write_known<<<148, 256>>>(p, N);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
    printf("GPU write 64 MB after CPU touch: %.2f ms = %.1f GB/s\n",
           ms, bytes * 2 / (ms/1000) / 1e9);

    free(p);
    return 0;
}
