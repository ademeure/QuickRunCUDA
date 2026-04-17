// PDL with realistic pattern: producer's blocks exit before signaling
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void producer_quick(unsigned long long *out, int cycles) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        while (clock64() - t0 < cycles) {}
        // After main work, signal dependents and exit
        asm volatile("griddepcontrol.launch_dependents;");
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

    int cycles = 5 * 2032 * 1000;  // 5 ms each kernel

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

    void *args[2] = {&d_out, &cycles};

    printf("# B300 PDL realistic: producer signals at end, then exits\n");
    printf("# Two 5ms kernels: with PDL, consumer starts as producer ends\n\n");

    // Without PDL
    {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(148); cfg.blockDim = dim3(32); cfg.stream = s;

        float t = bench([&]{
            cudaLaunchKernelExC(&cfg, (const void*)producer_quick, args);
            cudaLaunchKernelExC(&cfg, (const void*)consumer, args);
        });
        printf("  Without PDL:                      %.2f ms (expected ~10ms)\n", t);
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
            cudaLaunchKernelExC(&cfg, (const void*)producer_quick, args);
            cudaLaunchKernelExC(&cfg, (const void*)consumer, args);
        });
        printf("  With PDL StreamSerialization:     %.2f ms\n", t);
    }

    return 0;
}
