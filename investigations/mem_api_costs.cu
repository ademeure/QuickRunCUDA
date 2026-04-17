// Measure various memory API costs on B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);

    const int N = 100;
    auto measure = [&](auto fn) {
        // warmup
        for (int i = 0; i < 3; i++) fn();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::micro>(t1-t0).count() / N;
    };

    printf("# B300 memory API costs (per call, mean of %d calls)\n\n", N);

    // cudaMalloc/cudaFree various sizes
    for (size_t sz : {(size_t)4096, (size_t)65536, (size_t)(1<<20), (size_t)(16<<20), (size_t)(256<<20)}) {
        float t = measure([&]{
            void *p;
            cudaMalloc(&p, sz);
            cudaFree(p);
        });
        printf("  cudaMalloc+Free %zu bytes:        %.2f us\n", sz, t);
    }
    printf("\n");

    // cudaMallocHost (pinned)
    for (size_t sz : {(size_t)4096, (size_t)65536, (size_t)(1<<20), (size_t)(16<<20)}) {
        float t = measure([&]{
            void *p;
            cudaMallocHost(&p, sz);
            cudaFreeHost(p);
        });
        printf("  cudaMallocHost+Free %zu bytes:    %.2f us\n", sz, t);
    }
    printf("\n");

    // cudaMallocAsync (memory pool)
    cudaStream_t s; cudaStreamCreate(&s);
    for (size_t sz : {(size_t)4096, (size_t)65536, (size_t)(1<<20), (size_t)(16<<20)}) {
        float t = measure([&]{
            void *p;
            cudaMallocAsync(&p, sz, s);
            cudaFreeAsync(p, s);
        });
        // Need to sync to make sure pool ops complete
        cudaStreamSynchronize(s);
        printf("  cudaMallocAsync+Free %zu bytes:   %.2f us\n", sz, t);
    }
    printf("\n");

    // cudaMemcpy small
    {
        void *d, *h;
        cudaMalloc(&d, 4096);
        cudaMallocHost(&h, 4096);
        float t_h2d = measure([&]{ cudaMemcpy(d, h, 4096, cudaMemcpyHostToDevice); });
        float t_d2h = measure([&]{ cudaMemcpy(h, d, 4096, cudaMemcpyDeviceToHost); });
        printf("  cudaMemcpy 4KB H2D: %.2f us\n", t_h2d);
        printf("  cudaMemcpy 4KB D2H: %.2f us\n", t_d2h);
        cudaFree(d); cudaFreeHost(h);
    }
    printf("\n");

    // cudaMemcpyAsync small
    {
        void *d, *h;
        cudaMalloc(&d, 4096);
        cudaMallocHost(&h, 4096);
        float t_h2d = measure([&]{
            cudaMemcpyAsync(d, h, 4096, cudaMemcpyHostToDevice, s);
            cudaStreamSynchronize(s);
        });
        printf("  cudaMemcpyAsync 4KB H2D + sync: %.2f us\n", t_h2d);
        cudaFree(d); cudaFreeHost(h);
    }

    // cudaMemset
    {
        void *d;
        cudaMalloc(&d, 1 << 20);
        float t_ms = measure([&]{ cudaMemset(d, 0, 1 << 20); });
        printf("  cudaMemset 1MB:            %.2f us\n", t_ms);
        cudaFree(d);
    }

    cudaStreamDestroy(s);
    return 0;
}
