#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void producer(unsigned long long *out, int cycles) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        while (clock64() - t0 < cycles) {}
        asm volatile("griddepcontrol.launch_dependents;");
        unsigned long long t1 = clock64();
        while (clock64() - t1 < cycles) {}
        out[0] = clock64();
    }
}

extern "C" __global__ void consumer(unsigned long long *out, int cycles) {
    asm volatile("griddepcontrol.wait;");
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        while (clock64() - t0 < cycles) {}
        out[1] = clock64();
    }
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);
    unsigned long long *d_out; cudaMalloc(&d_out, 16 * sizeof(unsigned long long));

    int cycles = 5 * 2032 * 1000;

    auto bench = [&](auto fn, int trials = 5) {
        for (int i = 0; i < 3; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("# B300 PDL programmatic stream serialization\n");
    printf("# Producer: ~5ms work, signal, ~5ms more work; Consumer: ~5ms work\n");
    printf("# No-PDL expected: 5+5+5 = 15ms (producer completes before consumer starts)\n");
    printf("# With PDL: consumer overlaps with producer's 2nd half = ~10ms\n\n");

    void *args[2] = {&d_out, &cycles};

    // Without PDL
    {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(148); cfg.blockDim = dim3(32); cfg.stream = s;

        float t = bench([&]{
            cudaLaunchKernelExC(&cfg, (const void*)producer, args);
            cfg.gridDim=dim3(74); cudaLaunchKernelExC(&cfg, (const void*)consumer, args);
        });
        printf("  Without PDL (sequential):       %.2f ms\n", t);
    }

    // With PDL
    {
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;

        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(148); cfg.blockDim = dim3(32); cfg.stream = s;
        cfg.numAttrs = 1; cfg.attrs = &attr;

        float t = bench([&]{
            cudaLaunchKernelExC(&cfg, (const void*)producer, args);
            cfg.gridDim=dim3(74); cudaLaunchKernelExC(&cfg, (const void*)consumer, args);
        });
        printf("  With PDL StreamSerialization:   %.2f ms\n", t);
    }

    return 0;
}
