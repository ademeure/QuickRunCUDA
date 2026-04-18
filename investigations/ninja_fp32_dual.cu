// NINJA FP32: try to push past 85.5% via dual-issue FFMA + IADD3
// Hypothesis: FMA pipe at 85.7% leaves 14% headroom; pair with IADD3 to fill it.
//
// On Hopper/Blackwell, SMSPs can dual-issue 1 FFMA + 1 IADD3 per cycle.
// If kernel mixes 50/50 FFMA+IADD3, both pipes saturate.

#include <cuda_runtime.h>
#include <cstdio>
#include <nvml.h>

#ifndef ITERS
#define ITERS 16384
#endif

// Match D1 baseline EXACTLY (62 TFLOPS proven)
extern "C" __launch_bounds__(256, 4) __global__ void ffma_pure(float *out, float a_in, int iters) {
    float a = a_in + threadIdx.x;
    float b = a + 1, c = a + 2;
    float r0=0.5f, r1=1.5f, r2=2.5f, r3=3.5f;
    float r4=4.5f, r5=5.5f, r6=6.5f, r7=7.5f;
    float s0=8.5f, s1=9.5f, s2=10.5f, s3=11.5f;
    float s4=12.5f, s5=13.5f, s6=14.5f, s7=15.5f;
    float t0=16.5f, t1=17.5f, t2=18.5f, t3=19.5f;
    float t4=20.5f, t5=21.5f, t6=22.5f, t7=23.5f;
    a = a_in + (threadIdx.x & 1) + 0.001f * (threadIdx.x >> 1);  // Force runtime val

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        r0 = r0*a+b; r1 = r1*a+c; r2 = r2*a+b; r3 = r3*a+c;
        r4 = r4*a+b; r5 = r5*a+c; r6 = r6*a+b; r7 = r7*a+c;
        s0 = s0*b+a; s1 = s1*b+c; s2 = s2*b+a; s3 = s3*b+c;
        s4 = s4*b+a; s5 = s5*b+c; s6 = s6*b+a; s7 = s7*b+c;
        t0 = t0*c+a; t1 = t1*c+b; t2 = t2*c+a; t3 = t3*c+b;
        t4 = t4*c+a; t5 = t5*c+b; t6 = t6*c+a; t7 = t7*c+b;
    }
    float sum = r0+r1+r2+r3+r4+r5+r6+r7+s0+s1+s2+s3+s4+s5+s6+s7+t0+t1+t2+t3+t4+t5+t6+t7;
    if (sum < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = sum;
}

extern "C" __launch_bounds__(256, 4) __global__ void ffma_with_int(float *out, float a_in, int iters) {
    float a = a_in + threadIdx.x;
    float b = a + 1, c = a + 2;
    float r0=0.5f, r1=1.5f, r2=2.5f, r3=3.5f, r4=4.5f, r5=5.5f, r6=6.5f, r7=7.5f;
    float s0=8.5f, s1=9.5f, s2=10.5f, s3=11.5f, s4=12.5f, s5=13.5f, s6=14.5f, s7=15.5f;
    unsigned u0=1, u1=2, u2=3, u3=4, u4=5, u5=6, u6=7, u7=8;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        // 16 FFMA + 16 IADD3 (interleaved)
        r0 = r0*a+b; u0 = u0 + u1 + i;
        r1 = r1*a+c; u1 = u1 + u2 + i;
        r2 = r2*a+b; u2 = u2 + u3 + i;
        r3 = r3*a+c; u3 = u3 + u4 + i;
        r4 = r4*a+b; u4 = u4 + u5 + i;
        r5 = r5*a+c; u5 = u5 + u6 + i;
        r6 = r6*a+b; u6 = u6 + u7 + i;
        r7 = r7*a+c; u7 = u7 + u0 + i;
        s0 = s0*b+a; u0 = u0 + u4;
        s1 = s1*b+c; u1 = u1 + u5;
        s2 = s2*b+a; u2 = u2 + u6;
        s3 = s3*b+c; u3 = u3 + u7;
        s4 = s4*b+a;
        s5 = s5*b+c;
        s6 = s6*b+a;
        s7 = s7*b+c;
    }
    float sum = r0+r1+r2+r3+r4+r5+r6+r7+s0+s1+s2+s3+s4+s5+s6+s7;
    unsigned usum = u0+u1+u2+u3+u4+u5+u6+u7;
    if (sum < -1e30f || usum == 0xdeadbeef) out[blockIdx.x*blockDim.x+threadIdx.x] = sum + (float)usum;
}

int main() {
    cudaSetDevice(0);
    nvmlInit();
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex(0, &dev);
    int blocks = 148*8, threads = 256;
    float *d_out; cudaMalloc(&d_out, 1<<24);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](auto launch, int n_ffma_per_iter, const char* label) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 20; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        unsigned mhz; nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz);
        long ffma = (long)blocks * threads * ITERS * n_ffma_per_iter;
        double tflops = ffma * 2.0 / (best/1000) / 1e12;
        double th = 148*128*2*(mhz/1000.0)/1000;
        printf("  %s: %.4f ms = %.2f TFLOPS = %.2f%% theoretical (%u MHz)\n",
               label, best, tflops, tflops/th*100, mhz);
    };

    // ffma_pure has 24 chains but compiler sees vars not loop-invariant → smaller register
    // Set blocks/SM = 4 explicitly (same as D1)
    int b4 = 148*4;
    bench([&]{ ffma_pure<<<b4, threads>>>(d_out, 1.5f, ITERS); }, 24, "pure FFMA blk=148*4");
    bench([&]{ ffma_with_int<<<b4, threads>>>(d_out, 1.5f, ITERS); }, 16, "FFMA+IADD3 blk=148*4");

    nvmlShutdown();
    return 0;
}
