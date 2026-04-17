// Managed memory CPU-GPU ping-pong
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void touch_managed(int *p, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) p[i] += 1;
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    size_t bytes = 16 * 1024 * 1024;
    int N = bytes / 4;

    int *m;
    cudaMallocManaged(&m, bytes);

    // Initial CPU touch
    for (int i = 0; i < N; i++) m[i] = 0;

    auto bench_cpu_to_gpu = [&]() {
        // Touch on CPU
        m[0] = 1;
        // First GPU touch
        auto t0 = std::chrono::high_resolution_clock::now();
        touch_managed<<<148, 256, 0, s>>>(m, N);
        cudaStreamSynchronize(s);
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::micro>(t1-t0).count();
    };

    auto bench_gpu_to_cpu = [&]() {
        // Run on GPU
        touch_managed<<<148, 256, 0, s>>>(m, N);
        cudaStreamSynchronize(s);
        // Touch on CPU
        auto t0 = std::chrono::high_resolution_clock::now();
        int sum = 0;
        for (int i = 0; i < N; i++) sum += m[i];
        auto t1 = std::chrono::high_resolution_clock::now();
        if (sum < -1) m[0] = sum;
        return std::chrono::duration<float, std::micro>(t1-t0).count();
    };

    printf("# B300 managed memory CPU↔GPU first-touch costs (16 MB)\n\n");

    // First time
    printf("  First CPU→GPU first-touch: %.2f us\n", bench_cpu_to_gpu());
    printf("  First GPU→CPU first-touch: %.2f us\n", bench_gpu_to_cpu());

    // Subsequent
    float cpu_to_gpu = 1e30f;
    for (int i = 0; i < 5; i++) {
        float t = bench_cpu_to_gpu();
        if (t < cpu_to_gpu) cpu_to_gpu = t;
    }
    printf("\n  Best CPU→GPU after warmup: %.2f us\n", cpu_to_gpu);

    float gpu_to_cpu = 1e30f;
    for (int i = 0; i < 5; i++) {
        float t = bench_gpu_to_cpu();
        if (t < gpu_to_cpu) gpu_to_cpu = t;
    }
    printf("  Best GPU→CPU after warmup: %.2f us\n", gpu_to_cpu);

    cudaFree(m);
    return 0;
}
