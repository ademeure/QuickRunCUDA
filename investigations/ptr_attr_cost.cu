// cudaPointerGetAttributes cost
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);

    void *d; cudaMalloc(&d, 4096);
    void *h; cudaMallocHost(&h, 4096);
    void *m; cudaMallocManaged(&m, 4096);
    int stack;
    int *heap = new int[1024];

    auto bench = [&](void *p, const char *name) {
        cudaPointerAttributes attr;
        // Warmup
        for (int i = 0; i < 5; i++) cudaPointerGetAttributes(&attr, p);

        float best = 1e30f;
        for (int i = 0; i < 1000; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cudaPointerGetAttributes(&attr, p);
            auto t1 = std::chrono::high_resolution_clock::now();
            float ns = std::chrono::duration<float, std::nano>(t1-t0).count();
            if (ns < best) best = ns;
        }
        printf("  %-25s %.0f ns\n", name, best);
    };

    printf("# B300 cudaPointerGetAttributes cost\n\n");

    bench(d, "cudaMalloc ptr");
    bench(h, "cudaMallocHost ptr");
    bench(m, "cudaMallocManaged ptr");
    bench(&stack, "stack ptr");
    bench(heap, "C++ new ptr");

    delete[] heap;
    return 0;
}
