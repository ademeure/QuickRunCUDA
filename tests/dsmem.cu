// DSMEM (Distributed Shared Memory) on B300 - cluster shared memory access
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cooperative_groups.h>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

namespace cg = cooperative_groups;

// DSMEM kernel: each block reads from peer block's shared memory
template<int CSIZE>
__global__ void __cluster_dims__(CSIZE,1,1) dsmem_test(int n_iters, float *out) {
    auto cluster = cg::this_cluster();
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int block_rank = cluster.block_rank();

    // Init local shmem
    if (tid < 1024) smem[tid] = (float)(block_rank * 1024 + tid);
    cluster.sync();

    // Read from peer block's shmem in cluster
    int peer_rank = (block_rank + 1) % CSIZE;
    float *peer_smem = (float*)cluster.map_shared_rank(smem, peer_rank);

    float acc = 0;
    for (int i = 0; i < n_iters; i++) {
        acc += peer_smem[(tid + i) & 1023];
    }

    if (tid == 0) out[blockIdx.x] = acc;
}

extern "C" __global__ void dsmem_2(int n, float *o) { /* dummy declaration */ }

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;

    printf("# B300 DSMEM (Distributed Shared Memory) bandwidth\n");
    printf("# Each block reads from peer block's shmem in cluster\n\n");

    float *d_out;
    CK(cudaMalloc(&d_out, sm_count * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreate(&s));

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

    int threads = 1024;
    int sm_bytes = 4096;  // 1024 floats

    // Sweep cluster size
    printf("# Cluster size sweep (1024 thr, 4 KB shmem, 100k iters)\n");
    printf("# %-10s %-12s %-12s\n", "cluster", "time_ms", "BW_GB/s");

    int n_iters = 100000;

    auto run_csize = [&](int csize, auto kfn) {
        int g = (sm_count / csize) * csize;
        float t = bench([&]{
            kfn<<<g, threads, sm_bytes, s>>>(n_iters, d_out);
        });
        // Total reads = g blocks × 1024 thr × n_iters × 4 bytes (each block reads from peer)
        size_t total_reads = (size_t)g * threads * n_iters * 4;
        float bw = total_reads / (t/1e3f) / 1e9f;
        printf("  %-10d %-12.3f %-12.1f\n", csize, t, bw);
    };

    run_csize(2, dsmem_test<2>);
    run_csize(4, dsmem_test<4>);
    run_csize(8, dsmem_test<8>);

    // Compare to local shmem read
    printf("\n# Reference: local shmem read (no cluster)\n");
    {
        // Reuse cluster_2 but accessing own block (no map_shared_rank)
    }

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
