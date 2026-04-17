// Compare <<<>>> launch vs cuLaunchKernelEx vs cudaLaunchKernel
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}
extern "C" __global__ void with_args(int a, int b, int c) { (void)a; (void)b; (void)c; }

int main() {
    cudaSetDevice(0);
    cuInit(0);

    cudaStream_t s; cudaStreamCreate(&s);
    CUstream cs = (CUstream)s;

    // Get function pointer for cuLaunchKernelEx
    CUfunction f_noop, f_args;
    {
        CUmodule m;
        // Use the runtime kernel directly
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, with_args);
        // Get module from function
        f_noop = nullptr;
        f_args = nullptr;
    }

    auto bench = [&](auto fn, int trials = 200) {
        for (int i = 0; i < 5; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaStreamSynchronize(s);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    auto bench_async = [&](auto fn, int trials = 1000) {
        for (int i = 0; i < 5; i++) fn();
        cudaStreamSynchronize(s);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < trials; i++) fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1-t0).count();
        cudaStreamSynchronize(s);
        return us / trials;
    };

    printf("# B300 launch API comparison\n\n");

    // Method 1: <<<>>> with no args
    printf("## Triple-chevron <<<>>>\n");
    {
        printf("  noop launch + sync:    %.2f us\n",
               bench([&]{ noop<<<1, 32, 0, s>>>(); }));
        printf("  noop launch (async):   %.2f us\n",
               bench_async([&]{ noop<<<1, 32, 0, s>>>(); }));
    }
    printf("\n## With 3 int args\n");
    {
        printf("  with_args + sync:      %.2f us\n",
               bench([&]{ with_args<<<1, 32, 0, s>>>(1, 2, 3); }));
        printf("  with_args (async):     %.2f us\n",
               bench_async([&]{ with_args<<<1, 32, 0, s>>>(1, 2, 3); }));
    }

    // Method 2: cudaLaunchKernel
    printf("\n## cudaLaunchKernel\n");
    {
        void *args[] = {};
        printf("  noop + sync:           %.2f us\n",
               bench([&]{ cudaLaunchKernel((void*)noop, dim3(1), dim3(32), args, 0, s); }));
        printf("  noop (async):          %.2f us\n",
               bench_async([&]{ cudaLaunchKernel((void*)noop, dim3(1), dim3(32), args, 0, s); }));
    }
    {
        int a=1, b=2, c=3;
        void *args[] = {&a, &b, &c};
        printf("  with_args + sync:      %.2f us\n",
               bench([&]{ cudaLaunchKernel((void*)with_args, dim3(1), dim3(32), args, 0, s); }));
        printf("  with_args (async):     %.2f us\n",
               bench_async([&]{ cudaLaunchKernel((void*)with_args, dim3(1), dim3(32), args, 0, s); }));
    }

    // Method 3: cudaLaunchKernelExC (config-based)
    printf("\n## cudaLaunchKernelExC (config + sub-attribs)\n");
    {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(1);
        cfg.blockDim = dim3(32);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = s;
        cfg.numAttrs = 0;
        cfg.attrs = nullptr;

        printf("  noop + sync:           %.2f us\n",
               bench([&]{ cudaLaunchKernelExC(&cfg, (const void*)noop, nullptr); }));
        printf("  noop (async):          %.2f us\n",
               bench_async([&]{ cudaLaunchKernelExC(&cfg, (const void*)noop, nullptr); }));
    }

    return 0;
}
