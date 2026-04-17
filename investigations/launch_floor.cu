// Find absolute minimum kernel runtime measurable
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void truly_empty() {}
extern "C" __global__ void single_thread() { if (threadIdx.x == 0) {} }
extern "C" __global__ void single_clock(unsigned long long *out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    }
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);
    unsigned long long *d_out; cudaMalloc(&d_out, sizeof(unsigned long long));
    cudaEvent_t e0, e1; cudaEventCreateWithFlags(&e0, cudaEventDefault);
    cudaEventCreate(&e1);

    auto bench_event = [&](auto launch, int trials = 100) {
        for (int i = 0; i < 5; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            cudaEventRecord(e0, s);
            launch();
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            if (ms > 0 && ms < best) best = ms;
        }
        return best * 1000;  // us
    };

    printf("# B300 minimum measurable kernel runtime (cudaEventElapsedTime)\n\n");

    {
        float t = bench_event([&]{ truly_empty<<<1, 1, 0, s>>>(); });
        printf("  Empty kernel (1 block × 1 thread):     %.3f us\n", t);
    }
    {
        float t = bench_event([&]{ single_thread<<<1, 32, 0, s>>>(); });
        printf("  Empty kernel (1 block × 32 threads):   %.3f us\n", t);
    }
    {
        float t = bench_event([&]{ truly_empty<<<148, 32, 0, s>>>(); });
        printf("  Empty kernel (148 blocks × 32 threads): %.3f us\n", t);
    }
    {
        float t = bench_event([&]{ single_clock<<<1, 1, 0, s>>>(d_out); });
        printf("  Single clock64 read kernel:            %.3f us\n", t);
        unsigned long long cyc; cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        printf("    (in-kernel clock-clock = %llu cyc = %.2f ns)\n", cyc, cyc/2.032);
    }

    return 0;
}
