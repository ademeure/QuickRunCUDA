// Find max cluster size on B300
#include <cuda_runtime.h>
#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template<int CSIZE>
__global__ void __cluster_dims__(CSIZE,1,1) k_cluster(int *out) {
    auto cluster = cg::this_cluster();
    if (threadIdx.x == 0) {
        out[blockIdx.x] = cluster.num_blocks();
    }
}

extern "C" __global__ void k_c2(int *o) {} extern "C" __global__ void k_c4(int *o) {}
extern "C" __global__ void k_c8(int *o) {} extern "C" __global__ void k_c16(int *o) {}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm = prop.multiProcessorCount;

    int *d_out;
    cudaMalloc(&d_out, sm * sizeof(int));
    cudaMemset(d_out, 0xFF, sm * sizeof(int));

    auto try_cluster = [&](int csize, auto fn) {
        cudaMemset(d_out, 0xFF, sm * sizeof(int));
        int g = (sm / csize) * csize;
        fn<<<g, 32>>>(d_out);
        cudaError_t r = cudaGetLastError();
        cudaDeviceSynchronize();
        cudaError_t r2 = cudaGetLastError();
        int h[16];
        cudaMemcpy(h, d_out, 4 * 4, cudaMemcpyDeviceToHost);
        printf("  cluster_dim=%-3d (grid=%d): launch=%s, sync=%s, num_blocks=%d\n",
               csize, g,
               r == cudaSuccess ? "OK" : cudaGetErrorString(r),
               r2 == cudaSuccess ? "OK" : cudaGetErrorString(r2),
               h[0]);
    };

    printf("# B300 max cluster size test\n");
    try_cluster(2, k_cluster<2>);
    try_cluster(4, k_cluster<4>);
    try_cluster(8, k_cluster<8>);
    try_cluster(16, k_cluster<16>);

    cudaFree(d_out);
    return 0;
}
