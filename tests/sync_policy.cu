// Test SynchronizationPolicy: Spin vs Yield vs Auto vs BlockingSync
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" __global__ void compute(float *out, int iters, int k) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + k * 0.00001f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * sizeof(float)));

    auto bench = [&](cudaStream_t s, auto fn, int trials=10) {
        for (int i = 0; i < 2; i++) { fn(); cudaStreamSynchronize(s); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaStreamSynchronize(s);
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    int iters = 100;  // very fast kernel - sync overhead dominates

    cudaSynchronizationPolicy policies[] = {
        cudaSyncPolicyAuto,
        cudaSyncPolicySpin,
        cudaSyncPolicyYield,
        cudaSyncPolicyBlockingSync
    };
    const char *names[] = {"Auto", "Spin", "Yield", "BlockingSync"};

    printf("# B300 SynchronizationPolicy comparison (very small kernel)\n");
    printf("# Kernel: %d blocks × %d threads × %d iters\n\n", blocks, threads, iters);

    for (int pi = 0; pi < 4; pi++) {
        cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

        cudaLaunchAttributeValue val;
        val.syncPolicy = policies[pi];
        CK(cudaStreamSetAttribute(s, cudaLaunchAttributeSynchronizationPolicy, &val));

        float t = bench(s, [&]{
            compute<<<blocks,threads,0,s>>>(d_out, iters, 0);
        });
        printf("  %-15s : %.2f us per kernel+sync\n", names[pi], t*1000);

        cudaStreamDestroy(s);
    }

    // Also test with cudaSetDeviceFlags
    printf("\n# Effect of cudaDeviceFlags (host-side blocking sync):\n");
    {
        cudaDeviceReset();
        cudaSetDevice(0);
        // Default
        cudaStream_t s; cudaStreamCreate(&s);
        float t_default = bench(s, [&]{ compute<<<blocks,threads,0,s>>>(d_out, iters, 0); });
        printf("  default device flags:    %.2f us\n", t_default*1000);
        cudaStreamDestroy(s);
    }
    {
        cudaDeviceReset();
        cudaSetDevice(0);
        cudaSetDeviceFlags(cudaDeviceScheduleSpin);
        cudaStream_t s; cudaStreamCreate(&s);
        float t_spin = bench(s, [&]{ compute<<<blocks,threads,0,s>>>(d_out, iters, 0); });
        printf("  cudaDeviceScheduleSpin:  %.2f us\n", t_spin*1000);
        cudaStreamDestroy(s);
    }
    {
        cudaDeviceReset();
        cudaSetDevice(0);
        cudaSetDeviceFlags(cudaDeviceScheduleYield);
        cudaStream_t s; cudaStreamCreate(&s);
        float t_yield = bench(s, [&]{ compute<<<blocks,threads,0,s>>>(d_out, iters, 0); });
        printf("  cudaDeviceScheduleYield: %.2f us\n", t_yield*1000);
        cudaStreamDestroy(s);
    }
    {
        cudaDeviceReset();
        cudaSetDevice(0);
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        cudaStream_t s; cudaStreamCreate(&s);
        float t_block = bench(s, [&]{ compute<<<blocks,threads,0,s>>>(d_out, iters, 0); });
        printf("  cudaDeviceScheduleBlockingSync: %.2f us\n", t_block*1000);
        cudaStreamDestroy(s);
    }

    cudaFree(d_out);
    return 0;
}
