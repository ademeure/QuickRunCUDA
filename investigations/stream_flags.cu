// cudaStream flags effects on concurrent execution
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void busy(unsigned long long *out, int idx, int cycles) {
    if (threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        while (clock64() - t0 < cycles) {}
        out[idx] = clock64();
    }
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out; cudaMalloc(&d_out, 16 * sizeof(unsigned long long));

    auto bench_concurrent = [&](unsigned flags, const char *name) {
        cudaStream_t s1, s2;
        cudaStreamCreateWithFlags(&s1, flags);
        cudaStreamCreateWithFlags(&s2, flags);

        int delay = 5 * 2032 * 1000;  // 5 ms

        // Warmup
        for (int i = 0; i < 3; i++) {
            busy<<<1, 32, 0, s1>>>(d_out, 0, delay);
            busy<<<1, 32, 0, s2>>>(d_out, 1, delay);
            cudaDeviceSynchronize();
        }

        // Submit on both streams - should run concurrently
        auto t0 = std::chrono::high_resolution_clock::now();
        busy<<<1, 32, 0, s1>>>(d_out, 0, delay);
        busy<<<1, 32, 0, s2>>>(d_out, 1, delay);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
        printf("  %-25s 2 streams concurrent: %.2f ms (concurrency: %.1fx)\n",
               name, ms, 10.0f / ms);

        cudaStreamDestroy(s1); cudaStreamDestroy(s2);
    };

    printf("# B300 stream flag effects on concurrent execution\n");
    printf("# Two 5ms kernels on separate streams — should run in 5 ms if concurrent\n\n");

    bench_concurrent(cudaStreamDefault, "Default (blocking)");
    bench_concurrent(cudaStreamNonBlocking, "NonBlocking");

    // Test interaction with default stream
    printf("\n## With null stream interaction:\n");
    {
        cudaStream_t s_def, s_nb;
        cudaStreamCreateWithFlags(&s_def, cudaStreamDefault);
        cudaStreamCreateWithFlags(&s_nb, cudaStreamNonBlocking);

        int delay = 5 * 2032 * 1000;

        // Run on null stream first
        auto t0 = std::chrono::high_resolution_clock::now();
        busy<<<1, 32, 0, 0>>>(d_out, 0, delay);  // null stream
        busy<<<1, 32, 0, s_def>>>(d_out, 1, delay);  // blocking - waits for null
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms_def = std::chrono::duration<float, std::milli>(t1-t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        busy<<<1, 32, 0, 0>>>(d_out, 0, delay);
        busy<<<1, 32, 0, s_nb>>>(d_out, 1, delay);  // non-blocking - runs concurrent
        cudaDeviceSynchronize();
        t1 = std::chrono::high_resolution_clock::now();
        float ms_nb = std::chrono::duration<float, std::milli>(t1-t0).count();

        printf("  null + Default stream:    %.2f ms  (serialized expected ~10)\n", ms_def);
        printf("  null + NonBlocking:       %.2f ms  (concurrent expected ~5)\n", ms_nb);

        cudaStreamDestroy(s_def);
        cudaStreamDestroy(s_nb);
    }

    return 0;
}
