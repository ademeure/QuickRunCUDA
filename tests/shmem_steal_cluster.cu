// Isolated test: cluster.sync() with stolen reserved
#include <cuda_runtime.h>
#include <cstdio>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void __cluster_dims__(2,1,1) k_steal_cluster_iso(int *check, unsigned int *peeked) {
    extern __shared__ char buf[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    auto cluster = cg::this_cluster();

    // Steal reserved
    unsigned int magic = 0xCAFE0000 | bid;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(offset), "r"(magic + i) : "memory");
    }
    __syncthreads();

    // Verify before cluster.sync
    int corrupt_before = 0;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        unsigned int val;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(offset));
        if (val != magic + i) corrupt_before++;
    }
    __syncthreads();

    // Cluster sync
    cluster.sync();

    // Verify after cluster.sync
    int corrupt_after = 0;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        unsigned int val;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(offset));
        if (val != magic + i) corrupt_after++;
        if (tid == 0 && val != magic + i)
            peeked[bid * 256 + i] = val;
    }

    if (tid == 0) {
        check[bid * 2] = corrupt_before;
        check[bid * 2 + 1] = corrupt_after;
    }
}

int main() {
    cudaSetDevice(0);

    int *d_check;
    unsigned int *d_peeked;
    cudaMalloc(&d_check, 64);
    cudaMalloc(&d_peeked, 16 * 256 * 4);
    cudaMemset(d_check, 0, 64);
    cudaMemset(d_peeked, 0, 16 * 256 * 4);

    cudaFuncSetAttribute((void*)k_steal_cluster_iso,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 4096);

    k_steal_cluster_iso<<<2, 32, 4096>>>(d_check, d_peeked);
    cudaError_t r = cudaGetLastError();
    cudaDeviceSynchronize();
    cudaError_t r2 = cudaGetLastError();
    printf("Launch result: %s, sync: %s\n",
           r == cudaSuccess ? "OK" : cudaGetErrorString(r),
           r2 == cudaSuccess ? "OK" : cudaGetErrorString(r2));

    int check[16];
    cudaMemcpy(check, d_check, 16, cudaMemcpyDeviceToHost);
    unsigned int peeked[16 * 256];
    cudaMemcpy(peeked, d_peeked, 16 * 256 * 4, cudaMemcpyDeviceToHost);

    for (int b = 0; b < 2; b++) {
        printf("Block %d: corruption before cluster.sync=%d, after=%d\n",
               b, check[b * 2], check[b * 2 + 1]);
        if (check[b * 2 + 1] > 0) {
            printf("  Corrupted words:\n");
            for (int i = 0; i < 256; i++) {
                if (peeked[b * 256 + i] != 0) {
                    printf("    [0x%03x] expected 0x%08x got 0x%08x\n",
                           i*4, 0xCAFE0000 + b + i, peeked[b * 256 + i]);
                }
            }
        }
    }

    cudaFree(d_check); cudaFree(d_peeked);
    return 0;
}
