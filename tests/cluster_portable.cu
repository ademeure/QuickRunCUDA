// Try cluster size > 8 with non-portable mode
#include <cuda_runtime.h>
#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

extern "C" __global__ void k_cluster_dyn(int *out) {
    auto cluster = cg::this_cluster();
    if (threadIdx.x == 0) {
        out[blockIdx.x] = cluster.num_blocks();
    }
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm = prop.multiProcessorCount;

    int *d_out;
    cudaMalloc(&d_out, sm * sizeof(int));

    cudaStream_t s; cudaStreamCreate(&s);

    // Enable non-portable cluster on the kernel function
    cudaError_t er = cudaFuncSetAttribute((void*)k_cluster_dyn,
                                          cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    printf("cudaFuncSetAttribute(NonPortableClusterSizeAllowed=1): %s\n",
           er == cudaSuccess ? "OK" : cudaGetErrorString(er));

    // Test cluster sizes via launch attribute (dynamic, not __cluster_dims__)
    for (int csize : {2, 4, 8, 16, 32}) {
        cudaMemset(d_out, 0xFF, sm * sizeof(int));

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = csize;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;

        int g = (sm / csize) * csize;
        cudaLaunchConfig_t cfg = {dim3(g), dim3(32), 0, s, attrs, 1};

        void *args[] = {&d_out};
        cudaError_t r = cudaLaunchKernelExC(&cfg, (void*)k_cluster_dyn, args);
        cudaDeviceSynchronize();
        cudaError_t r2 = cudaGetLastError();

        int h;
        cudaMemcpy(&h, d_out, 4, cudaMemcpyDeviceToHost);
        printf("  cluster=%-3d non-portable: launch=%s, sync=%s, num_blocks=%d\n",
               csize,
               r == cudaSuccess ? "OK" : cudaGetErrorString(r),
               r2 == cudaSuccess ? "OK" : cudaGetErrorString(r2),
               h);
    }

    // Also try max device cluster size attribute
    int max_cluster_size;
    cudaDeviceGetAttribute(&max_cluster_size, cudaDevAttrClusterLaunch, 0);
    printf("\n# cudaDevAttrClusterLaunch: %d\n", max_cluster_size);

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
