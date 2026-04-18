// Persistent FFMA using PROVEN recipe from fp32_peak_honest.cu (74.5 TFLOPS catalog):
// 1024 thr/block, 2 blocks/SM (full occupancy = 32 warps/SM), ILP=16,
// inline asm fma.rn.f32, runtime constants from in_bc[]
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>
#include <numeric>

#define ILP 16

extern "C" __launch_bounds__(1024, 2) __global__ void k_persist(
    float *in_bc, float *out, volatile int *stop)
{
    float b = in_bc[0];
    float c = in_bc[1];
    float a[ILP];
    #pragma unroll
    for (int j = 0; j < ILP; j++) a[j] = in_bc[2 + j] + threadIdx.x * 0.0001f;

    long long n_outer = 0;
    while (!*stop) {
        #pragma unroll 1
        for (int i = 0; i < 256; i++) {  // shorter inner so n_outer increments more
            #pragma unroll
            for (int j = 0; j < ILP; j++)
                asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(a[j]) : "f"(b), "f"(c));
        }
        n_outer++;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float s = 0;
        #pragma unroll
        for (int j = 0; j < ILP; j++) s += a[j];
        out[0] = s + (float)n_outer;
        out[1] = (float)n_outer;
    }
}

int main() {
    cudaSetDevice(0);
    nvmlInit();
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex(0, &dev);

    int *stop_flag;
    cudaHostAlloc(&stop_flag, sizeof(int), cudaHostAllocMapped);
    *stop_flag = 0;
    int *d_stop; cudaHostGetDevicePointer(&d_stop, stop_flag, 0);

    int blocks = 148 * 2, threads = 1024;
    float h_bc[2 + ILP] = {1.00001f, 0.00001f};
    for (int i = 0; i < ILP; i++) h_bc[2+i] = 1.0f + i*0.1f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, sizeof(h_bc));
    cudaMalloc(&d_out, 1024);
    cudaMemcpy(d_in, h_bc, sizeof(h_bc), cudaMemcpyHostToDevice);

    cudaStream_t s; cudaStreamCreate(&s);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    std::atomic<bool> done{false};
    std::vector<unsigned> w, mhz;
    std::thread sampler([&]() {
        while (!done) {
            unsigned x;
            if (nvmlDeviceGetPowerUsage(dev, &x) == NVML_SUCCESS) w.push_back(x);
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &x) == NVML_SUCCESS) mhz.push_back(x);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    cudaEventRecord(e0, s);
    k_persist<<<blocks, threads, 0, s>>>(d_in, d_out, d_stop);
    std::this_thread::sleep_for(std::chrono::seconds(15));
    *stop_flag = 1;
    __sync_synchronize();
    cudaStreamSynchronize(s);
    cudaEventRecord(e1, s); cudaEventSynchronize(e1);
    done = true; sampler.join();

    float ms; cudaEventElapsedTime(&ms, e0, e1);
    float n_outer; cudaMemcpy(&n_outer, &d_out[1], 4, cudaMemcpyDeviceToHost);

    long total_ffma = (long)blocks * threads * (long)n_outer * 256 * ILP;
    double tflops = total_ffma * 2.0 / (ms/1000) / 1e12;
    auto pmax = *std::max_element(w.begin(), w.end());
    auto pavg = (unsigned)(std::accumulate(w.begin(), w.end(), 0ull) / w.size());
    auto mhzmin = *std::min_element(mhz.begin(), mhz.end());
    auto mhzmax = *std::max_element(mhz.begin(), mhz.end());

    printf("# Persistent FFMA proven recipe (1024 thr × 2 blk/SM = full occupancy)\n");
    printf("  blocks=%d threads=%d ILP=%d\n", blocks, threads, ILP);
    printf("  n_outer=%.0f, wall=%.2fs\n", n_outer, ms/1000);
    printf("  TFLOPS sustained: %.2f (theoretical at 1920 MHz: 72.74; at 2032 MHz: 76.96)\n", tflops);
    printf("  Power: avg %u, max %u W\n", pavg/1000, pmax/1000);
    printf("  Clock: min %u, max %u MHz\n", mhzmin, mhzmax);

    nvmlShutdown();
    return 0;
}
