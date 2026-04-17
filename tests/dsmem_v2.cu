// DSMEM (Distributed Shared Memory) bandwidth on B300
// Each block reads from peer block's shmem in same cluster
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define ITERS 1000

template<int CSIZE>
__global__ void __cluster_dims__(CSIZE,1,1) k_dsmem_read(float *out, int n_floats) {
    auto cluster = cg::this_cluster();
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int rank = cluster.block_rank();

    // Initialize own shmem
    for (int i = tid; i < n_floats; i += blockDim.x)
        smem[i] = (float)(rank * 1000 + i);
    cluster.sync();

    // Read from peer block (next in cluster)
    int peer = (rank + 1) % CSIZE;
    float *peer_smem = (float*)cluster.map_shared_rank(smem, peer);

    float acc = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        acc += peer_smem[(tid + i) & (n_floats - 1)];
    }

    cluster.sync();
    if (tid == 0 && acc < 1e30f) out[blockIdx.x] = acc;
}

template<int CSIZE>
__global__ void __cluster_dims__(CSIZE,1,1) k_local_smem_read(float *out, int n_floats) {
    auto cluster = cg::this_cluster();
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int rank = cluster.block_rank();

    for (int i = tid; i < n_floats; i += blockDim.x)
        smem[i] = (float)(rank * 1000 + i);
    cluster.sync();

    // Read from OWN shmem
    float acc = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        acc += smem[(tid + i) & (n_floats - 1)];
    }

    cluster.sync();
    if (tid == 0 && acc < 1e30f) out[blockIdx.x] = acc;
}

extern "C" __global__ void k_dsmem2(float *out, int n) { /* dummy decl */ }
extern "C" __global__ void k_dsmem4(float *out, int n) { /* dummy decl */ }
extern "C" __global__ void k_dsmem8(float *out, int n) { /* dummy decl */ }
extern "C" __global__ void k_local2(float *out, int n) { /* dummy decl */ }
extern "C" __global__ void k_local4(float *out, int n) { /* dummy decl */ }
extern "C" __global__ void k_local8(float *out, int n) { /* dummy decl */ }

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm = prop.multiProcessorCount;

    int threads = 256;
    int n_floats = 1024;  // 4 KB shmem per block
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

    printf("# B300 DSMEM (cluster shared memory) bandwidth\n");
    printf("# %d threads/block, %d shmem (%d floats), %d iters per thread\n\n",
           threads, shmem_bytes, n_floats, ITERS);
    printf("# %-10s %-15s %-15s %-10s\n", "cluster", "local_BW_GB/s", "remote_BW_GB/s", "ratio");

    auto run = [&](int csize, auto local_fn, auto remote_fn) {
        int g = (sm / csize) * csize;
        float t_local = bench([&]{
            local_fn<<<g, threads, shmem_bytes, s>>>(d_out, n_floats);
        });
        float t_remote = bench([&]{
            remote_fn<<<g, threads, shmem_bytes, s>>>(d_out, n_floats);
        });
        size_t total_bytes = (size_t)g * threads * ITERS * 4;
        float bw_local = total_bytes / (t_local/1e3) / 1e9;
        float bw_remote = total_bytes / (t_remote/1e3) / 1e9;
        printf("  %-10d %-15.1f %-15.1f %.2fx\n", csize, bw_local, bw_remote, bw_local/bw_remote);
    };

    run(2, k_local_smem_read<2>, k_dsmem_read<2>);
    run(4, k_local_smem_read<4>, k_dsmem_read<4>);
    run(8, k_local_smem_read<8>, k_dsmem_read<8>);
    run(16, k_local_smem_read<16>, k_dsmem_read<16>);

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
