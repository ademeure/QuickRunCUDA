// Cost of CUDA error-checking calls
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);
    cudaDeviceSynchronize();

    auto bench = [&](auto fn, int trials = 1000) {
        for (int i = 0; i < 5; i++) fn();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ns = std::chrono::duration<float, std::nano>(t1-t0).count();
            if (ns < best) best = ns;
        }
        return best;
    };

    printf("# B300 CUDA error checking call costs (idle context)\n\n");

    {
        float t = bench([&]{ cudaGetLastError(); });
        printf("  cudaGetLastError():        %.0f ns\n", t);
    }
    {
        float t = bench([&]{ cudaPeekAtLastError(); });
        printf("  cudaPeekAtLastError():     %.0f ns\n", t);
    }
    {
        float t = bench([&]{ cudaGetErrorString(cudaSuccess); });
        printf("  cudaGetErrorString():      %.0f ns\n", t);
    }
    {
        int dev;
        float t = bench([&]{ cudaGetDevice(&dev); });
        printf("  cudaGetDevice():           %.0f ns\n", t);
    }
    {
        int v;
        float t = bench([&]{ cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); });
        printf("  cudaDeviceGetAttribute():  %.0f ns\n", t);
    }
    {
        cudaError_t err;
        float t = bench([&]{ err = cudaStreamQuery(s); (void)err; });
        printf("  cudaStreamQuery() (idle):  %.0f ns\n", t);
    }
    return 0;
}
