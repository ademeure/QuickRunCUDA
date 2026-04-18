// D1 RIGOR: FP32 FFMA peak — find the 2-3% gap to theoretical
// THEORETICAL: 148 SMs × 128 FP32 cores × 2 op/FMA × 2.032 GHz = 76.96 TFLOPS
// CURRENT BEST: 74.6 TFLOPS (96.9%)
// HYPOTHESES:
//   H1: clock not actually at 2032 (only ~1990 sustained?)
//   H2: scheduling bubbles (FFMA pipe not 100% full)
//   H3: branch overhead in inner loop
//   H4: power-clamp throttling
//
// Test: vary ILP and chain count; record best wall-clock + ncu pipe metrics

#include <cuda_runtime.h>
#include <cstdio>
#include <nvml.h>

#ifndef CHAINS
#define CHAINS 8
#endif
#ifndef ILP
#define ILP 24
#endif
#ifndef ITERS
#define ITERS 8192
#endif

extern "C" __launch_bounds__(256, 6) __global__ void ffma_peak(float *out, float a_in, int iters) {
    float a = a_in + threadIdx.x;
    float b = a + 1.0f;
    float c = a + 2.0f;
    float d = a + 3.0f;
    float e = a + 4.0f;
    float f = a + 5.0f;
    float g = a + 6.0f;
    float h = a + 7.0f;

    float r0 = 0.5f, r1 = 1.5f, r2 = 2.5f, r3 = 3.5f;
    float r4 = 4.5f, r5 = 5.5f, r6 = 6.5f, r7 = 7.5f;
    float s0 = 8.5f, s1 = 9.5f, s2 = 10.5f, s3 = 11.5f;
    float s4 = 12.5f, s5 = 13.5f, s6 = 14.5f, s7 = 15.5f;
    float t0 = 16.5f, t1 = 17.5f, t2 = 18.5f, t3 = 19.5f;
    float t4 = 20.5f, t5 = 21.5f, t6 = 22.5f, t7 = 23.5f;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        // 24-way ILP, 3 chains × 8 vars
        r0 = r0 * a + b;  r1 = r1 * a + c;  r2 = r2 * a + d;  r3 = r3 * a + e;
        r4 = r4 * a + f;  r5 = r5 * a + g;  r6 = r6 * a + h;  r7 = r7 * a + b;
        s0 = s0 * b + a;  s1 = s1 * b + c;  s2 = s2 * b + d;  s3 = s3 * b + e;
        s4 = s4 * b + f;  s5 = s5 * b + g;  s6 = s6 * b + h;  s7 = s7 * b + a;
        t0 = t0 * c + a;  t1 = t1 * c + b;  t2 = t2 * c + d;  t3 = t3 * c + e;
        t4 = t4 * c + f;  t5 = t5 * c + g;  t6 = t6 * c + h;  t7 = t7 * c + a;
    }
    float sum = (r0+r1+r2+r3+r4+r5+r6+r7) + (s0+s1+s2+s3+s4+s5+s6+s7) + (t0+t1+t2+t3+t4+t5+t6+t7);
    if (sum < -1e30f) out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

int main() {
    cudaSetDevice(0);
    nvmlInit();
    nvmlDevice_t nvml_dev;
    nvmlDeviceGetHandleByIndex(0, &nvml_dev);

    int blocks = 148 * 8, threads = 256;  // 8 blk/SM × 256 thr = 2048 thr/SM (full occ)
    float *d_out; cudaMalloc(&d_out, blocks * threads * sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    for (int i = 0; i < 5; i++) ffma_peak<<<blocks, threads>>>(d_out, 1.5f, ITERS);
    cudaDeviceSynchronize();

    // Sample clock during run
    unsigned int clock_mhz_start, clock_mhz_end;
    nvmlDeviceGetClockInfo(nvml_dev, NVML_CLOCK_SM, &clock_mhz_start);

    float best = 1e30f;
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(e0);
        ffma_peak<<<blocks, threads>>>(d_out, 1.5f, ITERS);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    nvmlDeviceGetClockInfo(nvml_dev, NVML_CLOCK_SM, &clock_mhz_end);

    long total_ffma = (long)blocks * threads * ITERS * 24;
    double tflops = total_ffma * 2.0 / (best/1000) / 1e12;
    double theoretical = 148 * 128 * 2 * 2.032; // GFLOPS @ 2032 MHz
    printf("# Clock SM: %u → %u MHz\n", clock_mhz_start, clock_mhz_end);
    printf("# Theoretical at 2032 MHz: %.1f TFLOPS\n", theoretical/1000);
    printf("# Best: %.4f ms, %.2f TFLOPS = %.2f%% of theoretical\n",
           best, tflops, tflops/(theoretical/1000)*100);
    printf("# Theoretical at %u MHz: %.2f TFLOPS\n",
           clock_mhz_end, 148*128*2*(clock_mhz_end/1000.0)/1000);
    printf("# Ratio to clock-corrected theoretical: %.2f%%\n",
           tflops / (148*128*2*(clock_mhz_end/1000.0)/1000) * 100);

    nvmlShutdown();
    return 0;
}
