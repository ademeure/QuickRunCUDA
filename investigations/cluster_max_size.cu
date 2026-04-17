#include <cuda_runtime.h>
#include <cstdio>

template<int N> __global__ void __cluster_dims__(N,1,1) cluster_test(unsigned *out) {
    unsigned smid; asm("mov.u32 %0, %%smid;" : "=r"(smid));
    if (threadIdx.x == 0) out[blockIdx.x] = smid;
}

int main() {
    cudaSetDevice(0);
    unsigned *d; cudaMalloc(&d, 256 * sizeof(unsigned));

    auto try_cluster = [&](auto kernel, int N) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        cudaMemset(d, 0xff, 256 * sizeof(unsigned));
        kernel<<<N, 32>>>(d);
        cudaError_t err = cudaDeviceSynchronize();
        printf("  cluster N=%d: %s", N, cudaGetErrorString(err));
        if (err == cudaSuccess) {
            unsigned smids[256];
            cudaMemcpy(smids, d, N * sizeof(unsigned), cudaMemcpyDeviceToHost);
            unsigned mn = smids[0], mx = smids[0];
            for (int i = 0; i < N; i++) {
                if (smids[i] < mn) mn = smids[i];
                if (smids[i] > mx) mx = smids[i];
            }
            printf(" (range %u to %u)", mn, mx);
        }
        printf("\n");
    };

    printf("# B300 maximum cluster size with NonPortableClusterSize\n\n");
    try_cluster(cluster_test<8>, 8);
    try_cluster(cluster_test<16>, 16);
    try_cluster(cluster_test<32>, 32);
    try_cluster(cluster_test<64>, 64);

    return 0;
}
