// cudaStreamAttachMemAsync — control managed memory stream visibility
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void touch(float *p, int N, float v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) p[i] += v;
}

int main() {
    cudaSetDevice(0);

    cudaStream_t s; cudaStreamCreate(&s);
    size_t bytes = 16 * 1024 * 1024;
    int N = bytes / 4;

    float *m;
    cudaMallocManaged(&m, bytes, cudaMemAttachGlobal);
    memset(m, 0, bytes);

    auto bench_attach = [&](unsigned flags, const char *name) {
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            // Ensure managed mem is on CPU
            cudaMemLocation cpu_loc = {cudaMemLocationTypeHost, 0};
            cudaMemPrefetchAsync(m, bytes, cpu_loc, 0, s);
            cudaStreamSynchronize(s);

            // Attach to stream
            auto t0 = std::chrono::high_resolution_clock::now();
            cudaStreamAttachMemAsync(s, m, 0, flags);
            cudaStreamSynchronize(s);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        printf("  %-30s %.2f us\n", name, best);
    };

    printf("# B300 cudaStreamAttachMemAsync cost\n\n");
    bench_attach(cudaMemAttachGlobal, "Global attach");
    bench_attach(cudaMemAttachHost, "Host attach");
    bench_attach(cudaMemAttachSingle, "Single (this stream only)");

    // Measure first GPU touch after attach
    cudaMemLocation cpu_loc = {cudaMemLocationTypeHost, 0};
    cudaMemPrefetchAsync(m, bytes, cpu_loc, 0, s);
    cudaStreamSynchronize(s);
    cudaStreamAttachMemAsync(s, m, 0, cudaMemAttachSingle);

    auto t0 = std::chrono::high_resolution_clock::now();
    touch<<<148, 256, 0, s>>>(m, N, 1.0f);
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
    printf("\n  First GPU touch after Attach (16 MB): %.2f ms\n", ms);

    cudaFree(m);
    return 0;
}
