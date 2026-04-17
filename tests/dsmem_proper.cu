// Properly measure DSMEM bandwidth with high-throughput pattern
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define ITERS 10000  // many iters to amortize launch overhead

extern "C" __global__ void __cluster_dims__(2,1,1) k_dsmem_throughput(float *out, int n_floats) {
    auto cluster = cg::this_cluster();
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int rank = cluster.block_rank();

    // Init local
    for (int i = tid; i < n_floats; i += blockDim.x)
        smem[i] = (float)(rank * 1000 + i);
    cluster.sync();

    // Read from peer with multiple loads per iter
    int peer = (rank + 1) % 2;
    float *peer_smem = (float*)cluster.map_shared_rank(smem, peer);

    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        int base = (i * 256) & (n_floats - 256);
        a0 += peer_smem[base + tid];
        a1 += peer_smem[base + tid + 32];
        a2 += peer_smem[base + tid + 64];
        a3 += peer_smem[base + tid + 96];
        a4 += peer_smem[base + tid + 128];
        a5 += peer_smem[base + tid + 160];
        a6 += peer_smem[base + tid + 192];
        a7 += peer_smem[base + tid + 224];
    }

    cluster.sync();
    if (tid == 0) {
        out[blockIdx.x] = a0+a1+a2+a3+a4+a5+a6+a7;
    }
}

extern "C" __global__ void __cluster_dims__(2,1,1) k_local_throughput(float *out, int n_floats) {
    auto cluster = cg::this_cluster();
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int rank = cluster.block_rank();

    for (int i = tid; i < n_floats; i += blockDim.x)
        smem[i] = (float)(rank * 1000 + i);
    cluster.sync();

    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        int base = (i * 256) & (n_floats - 256);
        a0 += smem[base + tid];
        a1 += smem[base + tid + 32];
        a2 += smem[base + tid + 64];
        a3 += smem[base + tid + 96];
        a4 += smem[base + tid + 128];
        a5 += smem[base + tid + 160];
        a6 += smem[base + tid + 192];
        a7 += smem[base + tid + 224];
    }

    cluster.sync();
    if (tid == 0) {
        out[blockIdx.x] = a0+a1+a2+a3+a4+a5+a6+a7;
    }
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm = prop.multiProcessorCount;

    int threads = 256;
    int n_floats = 4096;  // 16 KB
    int shmem_bytes = n_floats * 4;

    float *d_out;
    cudaMalloc(&d_out, sm * sizeof(float));

    cudaStream_t s; cudaStreamCreate(&s);

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

    int g = (sm / 2) * 2;  // cluster=2

    float t_local = bench([&]{ k_local_throughput<<<g, threads, shmem_bytes, s>>>(d_out, n_floats); });
    float t_remote = bench([&]{ k_dsmem_throughput<<<g, threads, shmem_bytes, s>>>(d_out, n_floats); });

    // Total reads = blocks × threads × ITERS × 8 reads/iter × 4 bytes
    long long total_bytes = (long long)g * threads * ITERS * 8 * 4;
    float bw_local = total_bytes / (t_local/1e3) / 1e9;
    float bw_remote = total_bytes / (t_remote/1e3) / 1e9;

    printf("# B300 DSMEM proper test (8 reads/iter, register acc)\n");
    printf("# %d blocks (cluster=2), %d threads, %d floats shmem (%d KB), %d iter × 8 reads\n",
           g, threads, n_floats, shmem_bytes/1024, ITERS);
    printf("# Total reads: %lld (~%.2f GB)\n", total_bytes/4, total_bytes/1e9);
    printf("\n  Local SMEM: %.4f ms = %.1f GB/s aggregate (%.1f GB/s/SM)\n",
           t_local, bw_local, bw_local/sm);
    printf("  DSMEM:      %.4f ms = %.1f GB/s aggregate (%.1f GB/s/SM)\n",
           t_remote, bw_remote, bw_remote/sm);
    printf("  Ratio (local/remote): %.2fx\n", bw_local/bw_remote);

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
