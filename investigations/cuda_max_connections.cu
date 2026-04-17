// CUDA_DEVICE_MAX_CONNECTIONS effect on stream concurrency
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include <cstdlib>

extern "C" __global__ void busy(unsigned long long *out, int idx, int cycles) {
    if (threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        while (clock64() - t0 < cycles) {}
        out[idx] = clock64();
    }
}

int main() {
    cudaSetDevice(0);

    const char *env_val = getenv("CUDA_DEVICE_MAX_CONNECTIONS");
    printf("# CUDA_DEVICE_MAX_CONNECTIONS env: %s\n", env_val ? env_val : "(unset)");

    int max_streams = 64;
    unsigned long long *d_out;
    cudaMalloc(&d_out, max_streams * sizeof(unsigned long long));

    std::vector<cudaStream_t> streams(max_streams);
    for (int i = 0; i < max_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    int delay = 5 * 2032 * 1000;  // 5 ms

    auto bench = [&](int N, int trials = 3) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < N; j++) busy<<<1, 32, 0, streams[j]>>>(d_out, j, delay);
            cudaDeviceSynchronize();
        }
        float best = 1e30f;
        for (int t = 0; t < trials; t++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < N; j++) busy<<<1, 32, 0, streams[j]>>>(d_out, j, delay);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("\n# %-12s %-12s %-15s\n", "n_streams", "wall_ms", "concurrency");
    for (int N : {1, 2, 4, 8, 16, 32, 64}) {
        float t = bench(N);
        float conc = (5.0f * N) / t;
        printf("  %-12d %-12.3f %-15.1fx\n", N, t, conc);
    }

    return 0;
}
