// Compare cudaMemset launch overhead vs equivalent user kernel
#include <cuda_runtime.h>
#include <cstdio>
__global__ void noop_kernel() {}
__global__ void w1_kernel(int *p) { p[blockIdx.x * blockDim.x + threadIdx.x] = 0; }

int main() {
    cudaSetDevice(0);
    int *d; cudaMalloc(&d, 1024 * sizeof(int));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    for (int i = 0; i < 5; i++) cudaMemset(d, 0, 4); cudaDeviceSynchronize();

    auto bench = [&](auto fn, const char* label) {
        float best = 1e30f;
        for (int i = 0; i < 50; i++) {
            cudaEventRecord(e0);
            for (int j = 0; j < 1000; j++) fn();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        printf("  %s: %.3f us per call\n", label, best * 1000.0 / 1000.0);
    };

    bench([&]{ cudaMemset(d, 0, 4); }, "cudaMemset    4 B");
    bench([&]{ cudaMemset(d, 0, 4096); }, "cudaMemset 4 KB");
    bench([&]{ cudaMemsetAsync(d, 0, 4, 0); }, "cudaMemsetAsync 4 B");
    bench([&]{ noop_kernel<<<1, 1>>>(); }, "<<<1,1>>> noop");
    bench([&]{ w1_kernel<<<1, 32>>>(d); }, "<<<1,32>>> 1 store");

    return 0;
}
