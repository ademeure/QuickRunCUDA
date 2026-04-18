// I2 RIGOR: full-occupancy uses LESS power than partial for same TFLOPS?
// Catalog claim: 361W full-occ vs 437W partial = 17% power reduction.
//
// Method: run FFMA at full and partial occupancy, sample power via NVML
// repeatedly during execution; average over 1+ second.

#include <cuda_runtime.h>
#include <cstdio>
#include <nvml.h>
#include <chrono>
#include <thread>
#include <atomic>

#ifndef ITERS
#define ITERS 50000
#endif

extern "C" __launch_bounds__(256, 8) __global__ void ffma_full(float *out, float a) {
    float r0=0.5f, r1=1.5f, r2=2.5f, r3=3.5f, r4=4.5f, r5=5.5f, r6=6.5f, r7=7.5f;
    float b = a + 1, c = a + 2;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        r0 = r0*a+b; r1 = r1*a+c; r2 = r2*a+b; r3 = r3*a+c;
        r4 = r4*a+b; r5 = r5*a+c; r6 = r6*a+b; r7 = r7*a+c;
    }
    float s = r0+r1+r2+r3+r4+r5+r6+r7;
    if (s < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = s;
}

extern "C" __launch_bounds__(256, 1) __global__ void ffma_partial(float *out, float a) {
    float r0=0.5f, r1=1.5f, r2=2.5f, r3=3.5f, r4=4.5f, r5=5.5f, r6=6.5f, r7=7.5f;
    float b = a + 1, c = a + 2;
    // 8x more iters since we have 1/8 the blocks
    #pragma unroll 1
    for (int i = 0; i < ITERS * 8; i++) {
        r0 = r0*a+b; r1 = r1*a+c; r2 = r2*a+b; r3 = r3*a+c;
        r4 = r4*a+b; r5 = r5*a+c; r6 = r6*a+b; r7 = r7*a+c;
    }
    float s = r0+r1+r2+r3+r4+r5+r6+r7;
    if (s < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = s;
}

int main() {
    cudaSetDevice(0);
    nvmlInit();
    nvmlDevice_t nvml_dev;
    nvmlDeviceGetHandleByIndex(0, &nvml_dev);

    float *d_out; cudaMalloc(&d_out, 1<<24);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](auto launch, int total_blocks, int per_block_iters, const char* label) {
        // Warmup + sample power during kernel
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();

        // Sample power while running
        std::atomic<bool> done{false};
        std::atomic<int> n_samples{0};
        std::atomic<long long> sum_mw{0};
        std::atomic<unsigned> last_clk{0};
        std::thread sampler([&]() {
            while (!done) {
                unsigned int mw;
                if (nvmlDeviceGetPowerUsage(nvml_dev, &mw) == NVML_SUCCESS) {
                    sum_mw += mw;
                    n_samples++;
                }
                unsigned clk;
                nvmlDeviceGetClockInfo(nvml_dev, NVML_CLOCK_SM, &clk);
                last_clk = clk;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });

        cudaEventRecord(e0);
        for (int i = 0; i < 50; i++) launch();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        done = true;
        sampler.join();
        float ms; cudaEventElapsedTime(&ms, e0, e1);

        long total_ffma = (long)total_blocks * 256 * per_block_iters * 8 * 50;
        double tflops = total_ffma * 2.0 / (ms/1000) / 1e12;
        double avg_w = (double)sum_mw / n_samples / 1000.0;

        printf("  %s: %d blocks × %d iters\n", label, total_blocks, per_block_iters);
        printf("    Time: %.1f ms, %.1f TFLOPS, %.0f W avg, %u MHz, samples=%d\n",
               ms, tflops, avg_w, last_clk.load(), n_samples.load());
    };

    bench([&]{ ffma_full<<<148*8, 256>>>(d_out, 1.5f); },
          148*8, ITERS, "FULL-occ ");
    bench([&]{ ffma_partial<<<148, 256>>>(d_out, 1.5f); },
          148, ITERS*8, "PARTIAL  ");

    nvmlShutdown();
    return 0;
}
