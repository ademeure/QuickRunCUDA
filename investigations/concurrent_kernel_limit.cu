// How many concurrent kernel launches can be in-flight simultaneously?
// Earlier we said 128 dispatch slots — let's verify
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>

extern "C" __global__ void busy_one_block(unsigned long long *out, int idx, int cycles) {
    if (threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        while (clock64() - t0 < cycles) {}
        out[idx] = clock64() - t0;
    }
}

int main() {
    cudaSetDevice(0);

    int max_streams = 200;
    unsigned long long *d_out;
    cudaMalloc(&d_out, max_streams * sizeof(unsigned long long));

    std::vector<cudaStream_t> streams;
    for (int i = 0; i < max_streams; i++) {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        streams.push_back(s);
    }

    // Measure: launch N single-block kernels, each running ~5ms
    // If hardware can run them concurrently, total wall time = ~5ms
    // If serialized, wall time = N × 5ms
    int delay = 5 * 2032 * 1000;  // 5ms

    printf("# B300 concurrent kernel limit (5ms single-block kernels)\n");
    printf("# %-12s %-15s %-15s %-15s\n",
           "n_streams", "wall_ms", "per_kernel_ms", "concurrency");

    for (int N : {1, 4, 16, 32, 64, 100, 128, 144, 148, 160, 200}) {
        if (N > max_streams) continue;

        // Warmup
        for (int i = 0; i < N; i++)
            busy_one_block<<<1, 32, 0, streams[i]>>>(d_out, i, delay);
        cudaDeviceSynchronize();

        // Measure
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++)
            busy_one_block<<<1, 32, 0, streams[i]>>>(d_out, i, delay);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
        float per = ms / N;
        float conc = (5.0f * N) / ms;  // ideal_serial / actual = concurrency factor
        printf("  %-12d %-15.2f %-15.4f %-15.1f\n", N, ms, per, conc);
    }

    return 0;
}
