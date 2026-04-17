// Verify cluster blocks land on same GPC
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

__device__ unsigned get_smid() { unsigned s; asm("mov.u32 %0, %%smid;" : "=r"(s)); return s; }

__global__ void __cluster_dims__(8,1,1) cluster_8(unsigned *out) {
    if (threadIdx.x == 0) {
        out[blockIdx.x] = get_smid();
    }
}

__global__ void __cluster_dims__(16,1,1) cluster_16(unsigned *out) {
    if (threadIdx.x == 0) {
        out[blockIdx.x] = get_smid();
    }
}

__global__ void no_cluster(unsigned *out) {
    if (threadIdx.x == 0) {
        out[blockIdx.x] = get_smid();
    }
}

int main() {
    cudaSetDevice(0);
    unsigned *d_out; cudaMalloc(&d_out, 256 * sizeof(unsigned));

    printf("# B300 cluster launch SM placement\n");
    printf("# B300: 148 SMs across (likely) 9 GPCs of 16 SMs each\n\n");

    for (int trial = 0; trial < 3; trial++) {
        cudaMemset(d_out, 0xff, 256 * sizeof(unsigned));
        cluster_8<<<8, 32>>>(d_out);
        cudaDeviceSynchronize();
        unsigned smids[8];
        cudaMemcpy(smids, d_out, 8 * sizeof(unsigned), cudaMemcpyDeviceToHost);
        printf("  cluster_8 trial %d SMs: ", trial);
        for (int i = 0; i < 8; i++) printf("%d ", smids[i]);
        printf("(min=%d max=%d range=%d)\n",
               *std::min_element(smids, smids+8),
               *std::max_element(smids, smids+8),
               *std::max_element(smids, smids+8) - *std::min_element(smids, smids+8));
    }

    printf("\n");
    for (int trial = 0; trial < 3; trial++) {
        cudaMemset(d_out, 0xff, 256 * sizeof(unsigned));
        cluster_16<<<16, 32>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  cluster_16: %s\n", cudaGetErrorString(err));
            break;
        }
        unsigned smids[16];
        cudaMemcpy(smids, d_out, 16 * sizeof(unsigned), cudaMemcpyDeviceToHost);
        printf("  cluster_16 trial %d SMs: ", trial);
        for (int i = 0; i < 16; i++) printf("%d ", smids[i]);
        printf("\n  (min=%d max=%d range=%d)\n",
               *std::min_element(smids, smids+16),
               *std::max_element(smids, smids+16),
               *std::max_element(smids, smids+16) - *std::min_element(smids, smids+16));
    }

    return 0;
}
