// Concurrent cudaMallocAsync from multiple host threads
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>

int main() {
    cudaSetDevice(0);

    auto bench_threads = [&](int n_threads, int allocs_per) {
        std::vector<std::thread> threads;
        std::atomic<long long> total_us{0};

        for (int t = 0; t < n_threads; t++) {
            threads.emplace_back([&, t]() {
                cudaSetDevice(0);
                cudaStream_t s; cudaStreamCreate(&s);
                std::vector<void*> ptrs;

                auto t0 = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < allocs_per; i++) {
                    void *p; cudaMallocAsync(&p, 64*1024, s);
                    ptrs.push_back(p);
                }
                cudaStreamSynchronize(s);
                auto t1 = std::chrono::high_resolution_clock::now();
                long long us = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
                total_us.fetch_add(us);

                for (auto p : ptrs) cudaFreeAsync(p, s);
                cudaStreamSynchronize(s);
                cudaStreamDestroy(s);
            });
        }
        for (auto &t : threads) t.join();

        return total_us.load() / n_threads;
    };

    printf("# B300 cudaMallocAsync from multiple host threads\n");
    printf("# %-12s %-12s %-15s %-15s\n",
           "n_threads", "alloc/thr", "avg_total_us", "us_per_alloc");

    int allocs_per = 1000;
    for (int n : {1, 2, 4, 8, 16}) {
        long long avg_us = bench_threads(n, allocs_per);
        printf("  %-12d %-12d %-15lld %-15.2f\n",
               n, allocs_per, avg_us, (double)avg_us/allocs_per);
    }

    return 0;
}
