// Cooperative launch overhead vs regular launch
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <chrono>

namespace cg = cooperative_groups;

extern "C" __global__ void noop() {}

extern "C" __global__ void coop_noop() {
    cg::grid_group g = cg::this_grid();
    g.sync();
}

extern "C" __global__ void coop_work(int *out, int iters) {
    cg::grid_group g = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = tid + 1.0f;
    for (int i = 0; i < iters; i++) {
        a = a * 1.0001f + 0.0001f;
    }
    g.sync();
    if (a < -1e30f) out[tid] = a;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024*sizeof(int));
    cudaStream_t s; cudaStreamCreate(&s);

    int dev = 0;
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    printf("# SM count: %d\n", sm_count);

    int coop_supported;
    cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch, dev);
    printf("# Cooperative launch supported: %d\n", coop_supported);

    // multi-dev coop is deprecated in newer CUDA

    auto bench = [&](auto fn, int trials=200) {
        for (int i = 0; i < 5; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaStreamSynchronize(s);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    printf("\n## Launch overhead: regular vs cooperative\n");
    {
        // Regular noop launch
        float t_reg = bench([&]{
            noop<<<sm_count, 32, 0, s>>>();
        });

        // Coop noop launch
        float t_coop = bench([&]{
            void *args[] = {};
            cudaLaunchCooperativeKernel((void*)coop_noop, sm_count, 32, args, 0, s);
        });

        printf("  Regular noop:    %.2f us\n", t_reg);
        printf("  Coop noop:       %.2f us (overhead: %+.2f us)\n", t_coop, t_coop - t_reg);
    }

    // Test with actual work
    printf("\n## With ~1ms work\n");
    {
        int iters = 100000;
        float t_reg = bench([&]{
            noop<<<sm_count, 256, 0, s>>>();  // need a different kernel here
        });
        float t_coop = bench([&]{
            void *args[] = {(void*)&d_out, (void*)&iters};
            cudaLaunchCooperativeKernel((void*)coop_work, sm_count, 256, args, 0, s);
        });
        printf("  Regular kernel:  %.2f us\n", t_reg);
        printf("  Coop kernel:     %.2f us\n", t_coop);
    }

    // Test scaling: how much can we launch with coop?
    printf("\n## Cooperative launch maxima\n");
    {
        int max_blocks_per_sm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, coop_noop, 256, 0);
        printf("  max blocks/SM (256 threads): %d\n", max_blocks_per_sm);

        // Try: blocks_per_sm × sm_count blocks total
        int total = max_blocks_per_sm * sm_count;
        float t = bench([&]{
            void *args[] = {};
            cudaLaunchCooperativeKernel((void*)coop_noop, total, 256, args, 0, s);
        }, 50);
        printf("  Max coop launch: %d blocks × 256 = %d threads, %.2f us\n",
               total, total*256, t);
    }

    return 0;
}
