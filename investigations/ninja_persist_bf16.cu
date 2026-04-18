// Persistent BF16 mma.sync kernel — sustained tensor core test
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>
#include <numeric>

__launch_bounds__(256, 4) __global__ void mma_until_stop(
    float *out, volatile int *stop_flag)
{
    int laneId = threadIdx.x & 31;
    unsigned a0 = laneId, a1 = laneId+1, a2 = laneId+2, a3 = laneId+3;
    unsigned b0 = laneId*7, b1 = laneId*11;
    float c0=0,c1=0,c2=0,c3=0;
    long long n_outer = 0;
    while (!*stop_flag) {
        #pragma unroll 1
        for (int j = 0; j < 65536; j++) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
        }
        n_outer++;
    }
    if (c0+c1+c2+c3 < -1e30f && threadIdx.x == 0 && blockIdx.x == 0) out[0] = c0+c1+c2+c3;
    if (threadIdx.x == 0 && blockIdx.x == 0) out[1] = (float)n_outer;
}

int main() {
    cudaSetDevice(0);
    nvmlInit();
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex(0, &dev);

    int *stop_flag;
    cudaHostAlloc(&stop_flag, sizeof(int), cudaHostAllocMapped);
    *stop_flag = 0;
    int *d_stop_flag;
    cudaHostGetDevicePointer(&d_stop_flag, stop_flag, 0);

    float *d_out; cudaMalloc(&d_out, 16);
    int blocks = 148*4, threads = 256;
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
    mma_until_stop<<<blocks, threads, 0, s>>>(d_out, d_stop_flag);
    std::this_thread::sleep_for(std::chrono::seconds(30));
    *stop_flag = 1;
    __sync_synchronize();
    cudaStreamSynchronize(s);
    cudaEventRecord(e1, s); cudaEventSynchronize(e1);
    done = true; sampler.join();

    float ms; cudaEventElapsedTime(&ms, e0, e1);
    float n_outer; cudaMemcpy(&n_outer, &d_out[1], 4, cudaMemcpyDeviceToHost);

    int warps = blocks * threads / 32;
    long total_mma = warps * (long)n_outer * 65536 * 16;
    long total_flops = total_mma * 16 * 8 * 16 * 2;  // m*n*k*2 per mma
    double tflops = total_flops / (ms/1000) / 1e12;

    auto pmin = *std::min_element(w.begin(), w.end());
    auto pmax = *std::max_element(w.begin(), w.end());
    auto pavg = (unsigned)(std::accumulate(w.begin(), w.end(), 0ull) / w.size());
    auto mhzmin = *std::min_element(mhz.begin(), mhz.end());
    auto mhzmax = *std::max_element(mhz.begin(), mhz.end());

    printf("# Persistent mma.sync BF16 for ~30 sec\n");
    printf("  Outer loops: %.0f\n", n_outer);
    printf("  Wall: %.2f sec\n", ms/1000);
    printf("  TFLOPS sustained: %.1f (catalog mma.sync peak 569)\n", tflops);
    printf("  Power: min=%u, avg=%u, max=%u W\n", pmin/1000, pavg/1000, pmax/1000);
    printf("  Clock: min=%u, max=%u MHz\n", mhzmin, mhzmax);

    nvmlShutdown();
    return 0;
}
