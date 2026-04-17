// DSMEM at cluster=16 (B300 max non-portable)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define ITERS 1000

extern "C" __global__ void k_dsmem_dyn(float *out, int n_floats, int csize) {
    auto cluster = cg::this_cluster();
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int rank = cluster.block_rank();

    for (int i = tid; i < n_floats; i += blockDim.x)
        smem[i] = (float)(rank * 1000 + i);
    cluster.sync();

    int peer = (rank + 1) % csize;
    float *peer_smem = (float*)cluster.map_shared_rank(smem, peer);

    float acc = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        acc += peer_smem[(tid + i) & (n_floats - 1)];
    }

    cluster.sync();
    if (tid == 0 && acc < 1e30f) out[blockIdx.x] = acc;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm = prop.multiProcessorCount;

    cudaFuncSetAttribute((void*)k_dsmem_dyn,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

    int threads = 256;
    int n_floats = 1024;
    int shmem_bytes = n_floats * 4;

    float *d_out;
    cudaMalloc(&d_out, sm * sizeof(float));

    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](int csize, int trials=10) {
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim.x = csize;
        attr.val.clusterDim.y = 1;
        attr.val.clusterDim.z = 1;

        int g = (sm / csize) * csize;
        cudaLaunchConfig_t cfg = {dim3(g), dim3(threads), (unsigned)shmem_bytes, s, &attr, 1};
        int n = n_floats, c = csize;
        void *args[] = {&d_out, &n, &c};

        // Warmup
        for (int i = 0; i < 2; i++) {
            cudaLaunchKernelExC(&cfg, (void*)k_dsmem_dyn, args);
        }
        cudaDeviceSynchronize();

        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cudaLaunchKernelExC(&cfg, (void*)k_dsmem_dyn, args);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return std::make_pair(best, g);
    };

    printf("# B300 DSMEM at all cluster sizes (with non-portable allowed)\n");
    printf("# %-10s %-12s %-15s\n", "cluster", "time_ms", "remote_BW_GB/s");
    for (int csize : {2, 4, 8, 16}) {
        auto [t, g] = bench(csize);
        size_t total_bytes = (size_t)g * threads * ITERS * 4;
        printf("  %-10d %-12.3f %-15.1f\n",
               csize, t, total_bytes / (t/1e3) / 1e9);
    }

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
