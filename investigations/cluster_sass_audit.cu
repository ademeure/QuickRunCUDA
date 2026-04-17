#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2,1,1) cluster_sync_loop(int *out, int iters) {
    auto cluster = cg::this_cluster();
    int v = threadIdx.x;
    for (int i = 0; i < iters; i++) {
        cluster.sync();
        v += i;
    }
    if (v == 0xdeadbeef) out[blockIdx.x] = v;
}

int main() {
    cudaSetDevice(0);
    int *d; cudaMalloc(&d, 16);
    cluster_sync_loop<<<2, 32>>>(d, 100);
    cudaDeviceSynchronize();
    return 0;
}
