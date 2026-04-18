// Simpler: SINGLE long kernel with mapped-memory stop signal
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>
#include <numeric>

__launch_bounds__(256, 4) __global__ void ffma_until_stop(
    float *out, float a_in, volatile int *stop_flag)
{
    float a = a_in + threadIdx.x * 0.001f;
    float b = a + 1, c = a + 2;
    float r0=0.5f,r1=1.5f,r2=2.5f,r3=3.5f,r4=4.5f,r5=5.5f,r6=6.5f,r7=7.5f;
    float s0=8.5f,s1=9.5f,s2=10.5f,s3=11.5f,s4=12.5f,s5=13.5f,s6=14.5f,s7=15.5f;
    float t0=16.5f,t1=17.5f,t2=18.5f,t3=19.5f,t4=20.5f,t5=21.5f,t6=22.5f,t7=23.5f;
    long long n_outer = 0;
    while (!*stop_flag) {
        #pragma unroll 1
        for (int j = 0; j < 65536; j++) {  // many inner iters before checking flag
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                r0 = r0*a+b; r1 = r1*a+c; r2 = r2*a+b; r3 = r3*a+c;
                r4 = r4*a+b; r5 = r5*a+c; r6 = r6*a+b; r7 = r7*a+c;
                s0 = s0*b+a; s1 = s1*b+c; s2 = s2*b+a; s3 = s3*b+c;
                s4 = s4*b+a; s5 = s5*b+c; s6 = s6*b+a; s7 = s7*b+c;
                t0 = t0*c+a; t1 = t1*c+b; t2 = t2*c+a; t3 = t3*c+b;
                t4 = t4*c+a; t5 = t5*c+b; t6 = t6*c+a; t7 = t7*c+b;
            }
        }
        n_outer++;
    }
    float sum = r0+r1+r2+r3+r4+r5+r6+r7+s0+s1+s2+s3+s4+s5+s6+s7+t0+t1+t2+t3+t4+t5+t6+t7;
    if (sum < -1e30f && threadIdx.x == 0 && blockIdx.x == 0) out[0] = (float)n_outer;
    if (threadIdx.x == 0 && blockIdx.x == 0) out[1] = (float)n_outer;  // record always
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

    float *d_out; cudaMalloc(&d_out, 1<<24);
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
    ffma_until_stop<<<blocks, threads, 0, s>>>(d_out, 1.5f, d_stop_flag);

    // Sleep 30 sec
    std::this_thread::sleep_for(std::chrono::seconds(30));
    *stop_flag = 1;
    __sync_synchronize();

    cudaStreamSynchronize(s);
    cudaEventRecord(e1, s); cudaEventSynchronize(e1);
    done = true; sampler.join();

    float ms; cudaEventElapsedTime(&ms, e0, e1);
    float n_outer; cudaMemcpy(&n_outer, &d_out[1], 4, cudaMemcpyDeviceToHost);

    long total_ffma = (long)blocks * threads * (long)n_outer * 65536 * 16 * 24;
    double tflops = total_ffma * 2.0 / (ms/1000) / 1e12;

    auto pmin = *std::min_element(w.begin(), w.end());
    auto pmax = *std::max_element(w.begin(), w.end());
    auto pavg = (unsigned)(std::accumulate(w.begin(), w.end(), 0ull) / w.size());
    auto mhzmin = *std::min_element(mhz.begin(), mhz.end());
    auto mhzmax = *std::max_element(mhz.begin(), mhz.end());

    printf("# Persistent FFMA kernel for ~30 sec\n");
    printf("  Outer loops: %.0f (per thread)\n", n_outer);
    printf("  Wall: %.2f sec\n", ms/1000);
    printf("  TFLOPS sustained: %.2f (theoretical 1920 MHz: 72.74)\n", tflops);
    printf("  Power: min=%u, avg=%u, max=%u W\n", pmin/1000, pavg/1000, pmax/1000);
    printf("  Clock: min=%u, max=%u MHz\n", mhzmin, mhzmax);

    nvmlShutdown();
    return 0;
}
