// Test 4 async engines on B300 — concurrent H2D/D2H/D2D copies
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("# B300 async engines: %d\n", prop.asyncEngineCount);

    size_t bytes = 64 << 20;  // 64 MB transfers
    int n_streams = 8;

    float *d_data;
    CK(cudaMalloc(&d_data, bytes * n_streams));  // big buffer

    float *h_pinned[8];
    for (int i = 0; i < n_streams; i++)
        CK(cudaMallocHost(&h_pinned[i], bytes));

    std::vector<cudaStream_t> ss(n_streams);
    for (int i = 0; i < n_streams; i++)
        CK(cudaStreamCreateWithFlags(&ss[i], cudaStreamNonBlocking));

    auto bench = [&](auto fn, int trials=5) {
        for (int i = 0; i < 2; i++) { fn(); cudaDeviceSynchronize(); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    // ===== Test 1: H2D parallel on N streams =====
    printf("\n## H2D copies parallel on N streams (%zu MB each)\n", bytes >> 20);
    printf("# %-10s %-12s %-12s %-12s\n", "n_streams", "time_ms", "BW_GB/s", "concurrency");

    for (int n : {1, 2, 3, 4, 6, 8}) {
        size_t per = bytes / n;
        float t = bench([&]{
            for (int i = 0; i < n; i++)
                cudaMemcpyAsync((char*)d_data + (size_t)i * per, h_pinned[i], per, cudaMemcpyHostToDevice, ss[i]);
        });
        float bw = bytes / (t / 1000.0f) / 1e9f;
        printf("  %-10d %-12.3f %-12.1f\n", n, t, bw);
    }

    // ===== Test 2: H2D + D2H parallel (bidirectional) =====
    printf("\n## H2D + D2H bidirectional parallel\n");
    {
        float t_h2d = bench([&]{
            cudaMemcpyAsync(d_data, h_pinned[0], bytes, cudaMemcpyHostToDevice, ss[0]);
        });
        float t_d2h = bench([&]{
            cudaMemcpyAsync(h_pinned[1], d_data, bytes, cudaMemcpyDeviceToHost, ss[0]);
        });
        float t_bi = bench([&]{
            cudaMemcpyAsync(d_data, h_pinned[0], bytes, cudaMemcpyHostToDevice, ss[0]);
            cudaMemcpyAsync(h_pinned[1], d_data + (bytes/4), bytes/2, cudaMemcpyDeviceToHost, ss[1]);
        });
        printf("  H2D alone: %.3f ms (%.1f GB/s)\n", t_h2d, bytes/(t_h2d/1e3)/1e9);
        printf("  D2H alone: %.3f ms (%.1f GB/s)\n", t_d2h, bytes/(t_d2h/1e3)/1e9);
        printf("  Bidir (H2D+D2H/2): %.3f ms (overlap saving %+.3f vs serial %.3f)\n",
               t_bi, (t_h2d + t_d2h/2) - t_bi, t_h2d + t_d2h/2);
    }

    // ===== Test 3: Memory pool allocation speed =====
    printf("\n## Memory pool allocation speed\n");
    {
        cudaMemPool_t pool;
        cudaDeviceGetDefaultMemPool(&pool, 0);

        // Pre-trim and warm
        cudaMemPoolTrimTo(pool, 0);

        // Pool alloc
        const int N = 1000;
        void *ptrs[N];
        // Warmup
        for (int i = 0; i < 100; i++) cudaMallocAsync(&ptrs[0], 4096, ss[0]);
        cudaStreamSynchronize(ss[0]);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) cudaMallocAsync(&ptrs[i], 4096, ss[0]);
        cudaStreamSynchronize(ss[0]);
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("  cudaMallocAsync 4KB: %.3f us/call\n",
               std::chrono::duration<float, std::micro>(t1-t0).count() / N);

        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) cudaFreeAsync(ptrs[i], ss[0]);
        cudaStreamSynchronize(ss[0]);
        t1 = std::chrono::high_resolution_clock::now();
        printf("  cudaFreeAsync 4KB:   %.3f us/call\n",
               std::chrono::duration<float, std::micro>(t1-t0).count() / N);

        // vs cudaMalloc (sync, expensive)
        void *p;
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) {
            cudaMalloc(&p, 4096);
            cudaFree(p);
        }
        t1 = std::chrono::high_resolution_clock::now();
        printf("  cudaMalloc+Free 4KB: %.3f us/call (alloc+free)\n",
               std::chrono::duration<float, std::micro>(t1-t0).count() / 100);
    }

    // ===== Cleanup =====
    for (int i = 0; i < n_streams; i++) {
        cudaFreeHost(h_pinned[i]);
        cudaStreamDestroy(ss[i]);
    }
    cudaFree(d_data);
    return 0;
}
