// cooperative_groups cluster operations
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>

namespace cg = cooperative_groups;

__global__ void __cluster_dims__(8,1,1) cluster_sync_test(unsigned long long *out, int iters) {
    auto cluster = cg::this_cluster();
    int bid = blockIdx.x;
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        cluster.sync();
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0 && bid == 0) out[0] = t1 - t0;
}

__global__ void __cluster_dims__(8,1,1) cluster_dsmem_test(unsigned long long *out, int iters) {
    extern __shared__ int smem[];
    auto cluster = cg::this_cluster();
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    smem[tid] = tid + bid;
    cluster.sync();

    // Cross-block read via DSMEM
    int *peer_smem = (int*)cluster.map_shared_rank(smem, (bid + 1) % 8);

    unsigned long long t0 = clock64();
    int sum = 0;
    for (int i = 0; i < iters; i++) {
        sum += peer_smem[tid & 31];
    }
    unsigned long long t1 = clock64();
    if (sum < -1 << 30) smem[0] = sum;  // anti-DCE
    if (tid == 0 && bid == 0) out[0] = t1 - t0;
}

__global__ void smem_local_test(unsigned long long *out, int iters) {
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    smem[tid] = tid;
    __syncthreads();

    unsigned long long t0 = clock64();
    int sum = 0;
    for (int i = 0; i < iters; i++) {
        sum += smem[tid & 31];
    }
    unsigned long long t1 = clock64();
    if (sum < -1 << 30) smem[0] = sum;
    if (tid == 0) out[0] = t1 - t0;
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out; cudaMalloc(&d_out, 16 * sizeof(unsigned long long));

    int iters = 10000;
    int threads = 256;

    printf("# B300 cluster + DSMEM cooperative_groups operations\n\n");

    cluster_sync_test<<<8, threads, 0>>>(d_out, iters);
    cudaDeviceSynchronize();
    {
        unsigned long long cyc; cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  cluster.sync (8 blocks):  %.1f cyc = %.1f ns\n", per, per/2.032);
    }

    smem_local_test<<<1, threads, threads*sizeof(int)>>>(d_out, iters);
    cudaDeviceSynchronize();
    {
        unsigned long long cyc; cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  Local SHMEM read loop:    %.1f cyc = %.1f ns\n", per, per/2.032);
    }

    cluster_dsmem_test<<<8, threads, threads*sizeof(int)>>>(d_out, iters);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        unsigned long long cyc; cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  Cross-block DSMEM read:   %.1f cyc = %.1f ns\n", per, per/2.032);
    } else {
        printf("  DSMEM test failed: %s\n", cudaGetErrorString(err));
    }

    return 0;
}
