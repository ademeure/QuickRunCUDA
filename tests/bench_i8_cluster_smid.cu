// I8: cluster handle - which SMs get assigned to same cluster?
// Each block reads %smid; log to C buffer indexed by clusterID + ctaID-in-cluster
#ifndef CSIZE
#define CSIZE 4
#endif

extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(CSIZE, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

    // Get cluster ID via PTX
    unsigned int clusterRank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(clusterRank));

    // For each cluster (gridDim/CSIZE clusters), record SM IDs in [cluster_idx*CSIZE + clusterRank]
    unsigned int clusterIdx = blockIdx.x / CSIZE;
    ((unsigned int*)C)[clusterIdx * CSIZE + clusterRank] = smid;
}
