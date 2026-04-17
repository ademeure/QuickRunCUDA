// Minimum kernel launch latency on B300
// Measure: shortest time from kernel launch to kernel execution
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" __global__ void noop() {
    // intentionally empty
}

extern "C" __global__ void noop_with_clock(unsigned long long *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long c;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(c));
        out[0] = c;
    }
}

int main() {
    CK(cudaSetDevice(0));

    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](auto fn, int trials=20) {
        for (int i = 0; i < 5; i++) { fn(); cudaDeviceSynchronize(); }
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

    printf("# B300 kernel launch latency (1 block × 1 thread, empty kernel)\n\n");

    // Method 1: Just sync
    float t_sync = bench([&]{
        noop<<<1, 1, 0, s>>>();
    });
    printf("  Single launch + sync: %.2f us\n", t_sync*1000);

    // Method 2: Event-based timing
    {
        for (int i = 0; i < 5; i++) {
            noop<<<1, 1, 0, s>>>();
            cudaDeviceSynchronize();
        }
        float best = 1e30f;
        for (int i = 0; i < 50; i++) {
            cudaEventRecord(e0, s);
            noop<<<1, 1, 0, s>>>();
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        printf("  Single launch (event timing, no sync): %.2f us (kernel time only)\n", best*1000);
    }

    // Method 3: GPU-side clock vs CPU-side
    {
        unsigned long long *d_out;
        cudaMalloc(&d_out, 8);
        cudaMemset(d_out, 0, 8);

        // CPU timer + GPU clock measurement
        // Get CPU time, launch, GPU records its clock, then check delta
        const int N = 100;
        unsigned long long h_clocks[N];
        long long cpu_times_ns[N];

        for (int i = 0; i < N; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            noop_with_clock<<<1, 1, 0, s>>>(d_out);
            cudaMemcpy(&h_clocks[i], d_out, 8, cudaMemcpyDeviceToHost);
            cpu_times_ns[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            (void)t0;
        }

        // %globaltimer is in nanoseconds. Compare to host wall clock
        // Diff = GPU launch time - CPU launch time
        long long min_diff = LLONG_MAX;
        for (int i = 0; i < N; i++) {
            long long diff = (long long)h_clocks[i] - cpu_times_ns[i];
            if (i == 0 || labs(diff) < labs(min_diff)) min_diff = diff;
            (void)diff;
        }

        // Just print the variance of differences
        printf("  GPU globaltimer values (ns) - first 5:\n");
        for (int i = 0; i < 5; i++)
            printf("    %llu\n", h_clocks[i]);

        cudaFree(d_out);
    }

    // Method 4: Different launch APIs
    cudaLaunchConfig_t cfg = {dim3(1), dim3(1), 0, s, nullptr, 0};
    float t_kernEx = bench([&]{
        cudaLaunchKernelExC(&cfg, (void*)noop, nullptr);
    });
    printf("  cudaLaunchKernelExC + sync: %.2f us\n", t_kernEx*1000);

    void *args[] = {nullptr};
    float t_kern = bench([&]{
        cudaLaunchKernel((void*)noop, dim3(1), dim3(1), nullptr, 0, s);
    });
    printf("  cudaLaunchKernel + sync:    %.2f us\n", t_kern*1000);

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaStreamDestroy(s);
    return 0;
}
