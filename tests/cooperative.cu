// Cooperative kernels: grid-wide synchronization via cooperative_groups
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

namespace cg = cooperative_groups;

// Cooperative kernel: alternating compute + grid.sync()
extern "C" __global__ void coop_kernel(float *out, int iters, int sync_count) {
    auto grid = cg::this_grid();
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;

    for (int s = 0; s < sync_count; s++) {
        for (int i = 0; i < iters; i++)
            asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
        grid.sync();
    }

    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

// Comparison: persistent kernel without grid.sync() (just threads)
extern "C" __global__ void persist_kernel(float *out, int iters, int sync_count) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    for (int s = 0; s < sync_count; s++) {
        for (int i = 0; i < iters; i++)
            asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    }
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

// Same work but via N separate kernel launches (no PDL)
extern "C" __global__ void plain_compute(float *out, int iters, int k) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + k * 0.00001f;
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;
    int blocks = sm_count, threads = 128;

    int coop_supported;
    cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch, 0);
    printf("# Cooperative launch supported: %d\n", coop_supported);
    printf("# (Multi-device cooperative launch was deprecated in CUDA 11+)\n");

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 2; i++) { fn(); cudaDeviceSynchronize(); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    int iters = 1000;

    // ===== Compare 3 ways to do N "phases" of compute with sync between =====
    printf("\n# %d phases of %d FFMA per phase, %d blocks × %d threads\n\n", 4, iters, blocks, threads);

    for (int n_phases : {4, 8, 16, 32}) {
        // Method 1: cooperative kernel with grid.sync()
        cudaLaunchAttribute coop_attr;
        coop_attr.id = cudaLaunchAttributeCooperative;
        coop_attr.val.cooperative = 1;
        cudaLaunchConfig_t cfg = {dim3(blocks), dim3(threads), 0, s, &coop_attr, 1};

        float t_coop = bench([&]{
            int it = iters, sc = n_phases;
            void *args[] = {&d_out, &it, &sc};
            cudaError_t r = cudaLaunchKernelExC(&cfg, (void*)coop_kernel, args);
            if (r != cudaSuccess) printf("ERR %s\n", cudaGetErrorString(r));
        });

        // Method 2: plain kernel doing all phases with no sync (just for compute baseline)
        float t_persist = bench([&]{
            int it = iters, sc = n_phases;
            persist_kernel<<<blocks, threads, 0, s>>>(d_out, it, sc);
        });

        // Method 3: N separate kernel launches (sync via stream)
        float t_chain = bench([&]{
            for (int k = 0; k < n_phases; k++)
                plain_compute<<<blocks, threads, 0, s>>>(d_out, iters, k);
        });

        printf("  phases=%-3d coop_grid_sync=%.3f ms, persist_no_sync=%.3f ms, kernel_chain=%.3f ms\n",
               n_phases, t_coop, t_persist, t_chain);
        printf("              coop overhead/phase: %.2f us, chain overhead/phase: %.2f us\n",
               (t_coop - t_persist) * 1000 / n_phases,
               (t_chain - t_persist) * 1000 / n_phases);
    }

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
