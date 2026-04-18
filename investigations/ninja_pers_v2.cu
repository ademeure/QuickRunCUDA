// Persistent FFMA v2: try more warps per SM via smaller register footprint
// Also try concurrent multi-kernel with smaller chains
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>
#include <numeric>

#ifndef ILP
#define ILP 24
#endif
#ifndef BLOCKS_PER_SM
#define BLOCKS_PER_SM 4
#endif

__launch_bounds__(256, BLOCKS_PER_SM) __global__ void ffma_persist(
    float *out, float a_in, volatile int *stop_flag)
{
    float a = a_in + threadIdx.x * 0.001f;
    float b = a + 1, c = a + 2;
#if ILP == 8
    float r0=0.5f,r1=1.5f,r2=2.5f,r3=3.5f,r4=4.5f,r5=5.5f,r6=6.5f,r7=7.5f;
#elif ILP == 16
    float r0=0.5f,r1=1.5f,r2=2.5f,r3=3.5f,r4=4.5f,r5=5.5f,r6=6.5f,r7=7.5f;
    float s0=8.5f,s1=9.5f,s2=10.5f,s3=11.5f,s4=12.5f,s5=13.5f,s6=14.5f,s7=15.5f;
#elif ILP == 24
    float r0=0.5f,r1=1.5f,r2=2.5f,r3=3.5f,r4=4.5f,r5=5.5f,r6=6.5f,r7=7.5f;
    float s0=8.5f,s1=9.5f,s2=10.5f,s3=11.5f,s4=12.5f,s5=13.5f,s6=14.5f,s7=15.5f;
    float t0=16.5f,t1=17.5f,t2=18.5f,t3=19.5f,t4=20.5f,t5=21.5f,t6=22.5f,t7=23.5f;
#elif ILP == 32
    float r0=0.5f,r1=1.5f,r2=2.5f,r3=3.5f,r4=4.5f,r5=5.5f,r6=6.5f,r7=7.5f;
    float s0=8.5f,s1=9.5f,s2=10.5f,s3=11.5f,s4=12.5f,s5=13.5f,s6=14.5f,s7=15.5f;
    float t0=16.5f,t1=17.5f,t2=18.5f,t3=19.5f,t4=20.5f,t5=21.5f,t6=22.5f,t7=23.5f;
    float u0=24.5f,u1=25.5f,u2=26.5f,u3=27.5f,u4=28.5f,u5=29.5f,u6=30.5f,u7=31.5f;
#endif
    long long n_outer = 0;
    while (!*stop_flag) {
        #pragma unroll 1
        for (int j = 0; j < 65536; j++) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
#if ILP >= 8
                r0=r0*a+b; r1=r1*a+c; r2=r2*a+b; r3=r3*a+c;
                r4=r4*a+b; r5=r5*a+c; r6=r6*a+b; r7=r7*a+c;
#endif
#if ILP >= 16
                s0=s0*b+a; s1=s1*b+c; s2=s2*b+a; s3=s3*b+c;
                s4=s4*b+a; s5=s5*b+c; s6=s6*b+a; s7=s7*b+c;
#endif
#if ILP >= 24
                t0=t0*c+a; t1=t1*c+b; t2=t2*c+a; t3=t3*c+b;
                t4=t4*c+a; t5=t5*c+b; t6=t6*c+a; t7=t7*c+b;
#endif
#if ILP >= 32
                u0=u0*a+b; u1=u1*a+c; u2=u2*a+b; u3=u3*a+c;
                u4=u4*a+b; u5=u5*a+c; u6=u6*a+b; u7=u7*a+c;
#endif
            }
        }
        n_outer++;
    }
    float sum = 0;
#if ILP >= 8
    sum += r0+r1+r2+r3+r4+r5+r6+r7;
#endif
#if ILP >= 16
    sum += s0+s1+s2+s3+s4+s5+s6+s7;
#endif
#if ILP >= 24
    sum += t0+t1+t2+t3+t4+t5+t6+t7;
#endif
#if ILP >= 32
    sum += u0+u1+u2+u3+u4+u5+u6+u7;
#endif
    if (sum < -1e30f && threadIdx.x == 0 && blockIdx.x == 0) out[0] = sum;
    if (threadIdx.x == 0 && blockIdx.x == 0) out[1] = (float)n_outer;
}

int main() {
    cudaSetDevice(0);
    nvmlInit();
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex(0, &dev);

    int *stop_flag;
    cudaHostAlloc(&stop_flag, sizeof(int), cudaHostAllocMapped);
    *stop_flag = 0;
    int *d_stop;
    cudaHostGetDevicePointer(&d_stop, stop_flag, 0);

    float *d_out; cudaMalloc(&d_out, 1<<24);
    int blocks = 148 * BLOCKS_PER_SM, threads = 256;
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
    ffma_persist<<<blocks, threads, 0, s>>>(d_out, 1.5f, d_stop);
    std::this_thread::sleep_for(std::chrono::seconds(15));
    *stop_flag = 1;
    __sync_synchronize();
    cudaStreamSynchronize(s);
    cudaEventRecord(e1, s); cudaEventSynchronize(e1);
    done = true; sampler.join();

    float ms; cudaEventElapsedTime(&ms, e0, e1);
    float n_outer; cudaMemcpy(&n_outer, &d_out[1], 4, cudaMemcpyDeviceToHost);

    long total_ffma = (long)blocks * threads * (long)n_outer * 65536 * 16 * ILP;
    double tflops = total_ffma * 2.0 / (ms/1000) / 1e12;

    auto pmax = *std::max_element(w.begin(), w.end());
    auto pavg = (unsigned)(std::accumulate(w.begin(), w.end(), 0ull) / w.size());
    auto mhzmin = *std::min_element(mhz.begin(), mhz.end());

    printf("ILP=%d BLK/SM=%d: TFLOPS=%.2f (=%.1f%% of 72.74), Power=%uW avg %uW max, clk min=%u\n",
           ILP, BLOCKS_PER_SM, tflops, tflops/72.74*100, pavg/1000, pmax/1000, mhzmin);

    nvmlShutdown();
    return 0;
}
