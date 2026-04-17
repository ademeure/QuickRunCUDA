// How many copy engines does B300 have? How much parallelism in DMA?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>

int main() {
    cudaSetDevice(0);

    int async_engines;
    cudaDeviceGetAttribute(&async_engines, cudaDevAttrAsyncEngineCount, 0);
    printf("# B300 AsyncEngineCount: %d\n", async_engines);

    size_t size_mb = 256;
    size_t bytes = size_mb * 1024 * 1024;

    void *h_buf;
    cudaError_t err = cudaMallocHost(&h_buf, bytes * 8);
    if (err) { printf("cudaMallocHost failed: %s\n", cudaGetErrorString(err)); return 1; }
    memset(h_buf, 0, bytes * 8);

    float *d_buf;
    err = cudaMalloc(&d_buf, bytes * 8);
    if (err) { printf("cudaMalloc failed: %s\n", cudaGetErrorString(err)); return 1; }

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

    printf("\n# Copy parallelism: N streams × %zu MB H2D\n", size_mb);
    printf("# %-12s %-15s %-15s %-15s\n", "n_streams", "time_ms", "agg_GB/s", "per-stream_GB/s");

    // Limit total bytes per launch to fit our buffer (8 chunks)
    for (int N : {1, 2, 4, 8}) {
        std::vector<cudaStream_t> streams(N);
        for (int i = 0; i < N; i++) cudaStreamCreate(&streams[i]);

        float t = bench([&]{
            for (int i = 0; i < N; i++)
                cudaMemcpyAsync((char*)d_buf + i*bytes, (char*)h_buf + i*bytes, bytes,
                                cudaMemcpyHostToDevice, streams[i]);
        });
        double gb = N * (double)bytes / 1e9;
        double bw = gb / (t/1000);
        printf("  %-12d %-15.2f %-15.1f %-15.1f\n", N, t, bw, bw/N);

        for (auto &s : streams) cudaStreamDestroy(s);
    }

    // H2D + D2H simultaneously (full-duplex test)
    printf("\n# Full-duplex: H2D and D2H on separate streams\n");
    {
        cudaStream_t s1, s2;
        cudaStreamCreate(&s1);
        cudaStreamCreate(&s2);

        float t_h2d = bench([&]{
            cudaMemcpyAsync(d_buf, h_buf, bytes, cudaMemcpyHostToDevice, s1);
        });
        float t_d2h = bench([&]{
            cudaMemcpyAsync(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost, s2);
        });
        float t_both = bench([&]{
            cudaMemcpyAsync(d_buf, h_buf, bytes, cudaMemcpyHostToDevice, s1);
            cudaMemcpyAsync((char*)h_buf + bytes, (char*)d_buf + bytes, bytes, cudaMemcpyDeviceToHost, s2);
        });

        float bw_h2d = bytes / (t_h2d/1000) / 1e9;
        float bw_d2h = bytes / (t_d2h/1000) / 1e9;
        float bw_both = 2.0f * bytes / (t_both/1000) / 1e9;

        printf("  H2D alone: %.2f ms = %.1f GB/s\n", t_h2d, bw_h2d);
        printf("  D2H alone: %.2f ms = %.1f GB/s\n", t_d2h, bw_d2h);
        printf("  Both together: %.2f ms (agg %.1f GB/s)\n", t_both, bw_both);
        printf("  Full-duplex efficiency: %.1f%% of sum\n",
               100.0 * bw_both / (bw_h2d + bw_d2h));

        cudaStreamDestroy(s1); cudaStreamDestroy(s2);
    }

    // D2D within same device
    printf("\n# D2D (same device) bandwidth\n");
    {
        cudaStream_t s; cudaStreamCreate(&s);
        float t = bench([&]{
            cudaMemcpyAsync(d_buf, (char*)d_buf + 4*bytes, bytes, cudaMemcpyDeviceToDevice, s);
        });
        printf("  D2D same device: %.2f ms = %.1f GB/s\n", t, bytes/(t/1000)/1e9);
        cudaStreamDestroy(s);
    }

    // Small-copy latency
    printf("\n# Small copy latency\n");
    printf("  %-10s %-12s %-12s\n", "size", "h2d_us", "d2h_us");
    for (size_t sz : {4ul, 1024ul, 65536ul, 1024ul*1024}) {
        cudaStream_t s; cudaStreamCreate(&s);
        float th = bench([&]{ cudaMemcpyAsync(d_buf, h_buf, sz, cudaMemcpyHostToDevice, s); }, 200);
        float td = bench([&]{ cudaMemcpyAsync(h_buf, d_buf, sz, cudaMemcpyDeviceToHost, s); }, 200);
        printf("  %-10zu %-12.2f %-12.2f\n", sz, th*1000, td*1000);
        cudaStreamDestroy(s);
    }

    cudaFree(d_buf); cudaFreeHost(h_buf);
    return 0;
}
