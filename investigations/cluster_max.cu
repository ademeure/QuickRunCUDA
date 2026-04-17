// Maximum cluster size with non-portable enabled
#include <cuda_runtime.h>
#include <cstdio>

__device__ unsigned get_smid() { unsigned s; asm("mov.u32 %0, %%smid;" : "=r"(s)); return s; }

__global__ void __cluster_dims__(8,1,1) cluster_8(unsigned *out) {
    if (threadIdx.x == 0) out[blockIdx.x] = get_smid();
}

__global__ void __cluster_dims__(16,1,1) cluster_16(unsigned *out) {
    if (threadIdx.x == 0) out[blockIdx.x] = get_smid();
}

int main() {
    cudaSetDevice(0);

    unsigned *d_out; cudaMalloc(&d_out, 256 * sizeof(unsigned));

    // Enable non-portable cluster size
    cudaError_t err = cudaFuncSetAttribute(cluster_16,
        cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    printf("# Set NonPortableClusterSize: %s\n", cudaGetErrorString(err));

    // Try cluster_16 again
    cudaMemset(d_out, 0xff, 256 * sizeof(unsigned));
    cluster_16<<<16, 32>>>(d_out);
    err = cudaDeviceSynchronize();
    printf("# cluster_16 sync: %s\n", cudaGetErrorString(err));

    if (err == cudaSuccess) {
        unsigned smids[16];
        cudaMemcpy(smids, d_out, 16 * sizeof(unsigned), cudaMemcpyDeviceToHost);
        printf("  cluster_16 SMs: ");
        for (int i = 0; i < 16; i++) printf("%d ", smids[i]);
        unsigned mn = smids[0], mx = smids[0];
        for (int i = 0; i < 16; i++) {
            if (smids[i] < mn) mn = smids[i];
            if (smids[i] > mx) mx = smids[i];
        }
        printf("\n  range: %u to %u (span = %u)\n", mn, mx, mx - mn);
    }

    // Try even larger (max 16 non-portable on B300?)
    int max_cluster_dim;
    cudaDeviceGetAttribute(&max_cluster_dim, cudaDevAttrClusterLaunch, 0);
    printf("\n# cudaDevAttrClusterLaunch: %d\n", max_cluster_dim);

    return 0;
}
