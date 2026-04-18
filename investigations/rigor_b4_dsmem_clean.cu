// B4 RIGOR v2: DSMEM peak read throughput, no inner sync, anti-DCE via XOR chain
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 4
#endif
#ifndef ITERS
#define ITERS 1024
#endif
#ifndef ILP
#define ILP 8
#endif

extern "C" __launch_bounds__(256, 1) __global__ void
__cluster_dims__(CLUSTER_SIZE, 1, 1)
dsmem_chain(unsigned *gout, int iters)
{
    extern __shared__ unsigned smem[];
    int tid = threadIdx.x;
    int rank = blockIdx.x % CLUSTER_SIZE;
    int peer = (rank + 1) % CLUSTER_SIZE;

    // Initialize local SMEM (will be read by other peers)
    #pragma unroll
    for (int i = 0; i < 8; i++) smem[tid + i*256] = (rank * 1024 + tid + i) ^ 0xa5a5a5a5;

    auto cluster = cg::this_cluster();
    cluster.sync();

    unsigned *peer_smem = cluster.map_shared_rank(smem, peer);
    unsigned acc = 0;

    // Tight loop: ILP independent loads per iter from PEER smem
    // Address depends on iter to defeat LICM
    #pragma unroll 1
    for (int it = 0; it < iters; it++) {
        unsigned base = (it * 7) & 255;
        unsigned r0 = peer_smem[(base + tid +   0) & 1023];
        unsigned r1 = peer_smem[(base + tid + 128) & 1023];
        unsigned r2 = peer_smem[(base + tid + 256) & 1023];
        unsigned r3 = peer_smem[(base + tid + 384) & 1023];
        unsigned r4 = peer_smem[(base + tid + 512) & 1023];
        unsigned r5 = peer_smem[(base + tid + 640) & 1023];
        unsigned r6 = peer_smem[(base + tid + 768) & 1023];
        unsigned r7 = peer_smem[(base + tid + 896) & 1023];
        acc ^= r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7;
    }

    cluster.sync();
    if (acc == 0xdeadbeef) gout[blockIdx.x * blockDim.x + tid] = acc;
    else if (tid == 0 && blockIdx.x == 0) gout[0] = acc;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    int iters = (argc > 1) ? atoi(argv[1]) : ITERS;
    // Saturate the GPU: launch enough clusters
    // 148 SMs / CLUSTER_SIZE = max simultaneous clusters
    int n_clusters = 148 / CLUSTER_SIZE;
    int blocks = n_clusters * CLUSTER_SIZE;
    int threads = 256;
    int smem_bytes = 2048 * sizeof(unsigned);  // 8 KB per block

    unsigned *d_out; cudaMalloc(&d_out, blocks * threads * sizeof(unsigned));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks, 1, 1);
    cfg.blockDim = dim3(threads, 1, 1);
    cfg.dynamicSmemBytes = smem_bytes;
    cudaFuncSetAttribute(dsmem_chain, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {CLUSTER_SIZE, 1, 1};
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    auto launch = [&]() {
        cudaLaunchKernelEx(&cfg, dsmem_chain, d_out, iters);
    };

    for (int i = 0; i < 5; i++) launch();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("ERR: %s\n", cudaGetErrorString(err)); return 1; }

    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        launch();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }

    long warps = blocks * (threads / 32);
    long bytes_dsmem = warps * (long)iters * ILP * 128;  // ILP loads × 32 lanes × 4 B
    double tbs_dsmem = bytes_dsmem / (best/1000) / 1e12;

    printf("# CL=%d, iters=%d, ILP=%d, blocks=%d, %.4f ms\n",
           CLUSTER_SIZE, iters, ILP, blocks, best);
    printf("  DSMEM read aggregate: %.3f TB/s\n", tbs_dsmem);
    printf("  Per-cluster DSMEM:    %.1f GB/s\n", tbs_dsmem * 1000 / n_clusters);
    printf("  Per-block DSMEM:      %.1f GB/s\n", tbs_dsmem * 1000 / blocks);

    return 0;
}
