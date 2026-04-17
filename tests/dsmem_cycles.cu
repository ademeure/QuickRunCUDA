// Measure DSMEM bandwidth via clock64 cycles per load
#include <cuda_runtime.h>
#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define N_LOADS 256

extern "C" __global__ void __cluster_dims__(2,1,1) dsmem_cy_test(unsigned long long *out, int *src_idx, int *dst_idx) {
    auto cluster = cg::this_cluster();
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int rank = cluster.block_rank();

    // Init local shmem (4 KB = 1024 floats)
    for (int i = tid; i < 1024; i += blockDim.x)
        smem[i] = (float)(tid + rank * 1000 + i);
    cluster.sync();

    // Get peer pointer
    int peer = (rank + 1) % 2;
    float *peer_smem = (float*)cluster.map_shared_rank(smem, peer);

    // Load INDICES from global so compiler can't predict
    int idx0 = src_idx[tid % 32] & 1023;

    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    // Pointer chase via dependent loads
    float val = peer_smem[idx0];
    int next = (int)val & 1023;
    #pragma unroll 1
    for (int i = 0; i < N_LOADS; i++) {
        val = peer_smem[next];
        next = (int)val & 1023;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));

    if (tid == 0 && rank == 0) {
        out[blockIdx.x] = end - start;
        dst_idx[blockIdx.x] = next;  // defeat DCE
    }
}

extern "C" __global__ void local_cy_test(unsigned long long *out, int *src_idx, int *dst_idx) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    for (int i = tid; i < 1024; i += blockDim.x)
        smem[i] = (float)(tid + i);
    __syncthreads();

    int idx0 = src_idx[tid % 32] & 1023;

    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    float val = smem[idx0];
    int next = (int)val & 1023;
    #pragma unroll 1
    for (int i = 0; i < N_LOADS; i++) {
        val = smem[next];
        next = (int)val & 1023;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));

    if (tid == 0) {
        out[blockIdx.x] = end - start;
        dst_idx[blockIdx.x] = next;
    }
}

int main() {
    cudaSetDevice(0);

    int *d_src, *d_dst;
    cudaMalloc(&d_src, 32 * sizeof(int));
    cudaMalloc(&d_dst, 256 * sizeof(int));
    int h_src[32];
    for (int i = 0; i < 32; i++) h_src[i] = i * 31;  // some pattern
    cudaMemcpy(d_src, h_src, 32 * 4, cudaMemcpyHostToDevice);

    unsigned long long *d_out;
    cudaMalloc(&d_out, 16 * sizeof(unsigned long long));

    cudaFuncSetAttribute((void*)dsmem_cy_test,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 4096);
    cudaFuncSetAttribute((void*)local_cy_test,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 4096);

    // DSMEM
    dsmem_cy_test<<<2, 32, 4096>>>(d_out, d_src, d_dst);
    cudaDeviceSynchronize();
    unsigned long long h;
    cudaMemcpy(&h, d_out, 8, cudaMemcpyDeviceToHost);
    double cy_per_dsmem = (double)h / N_LOADS;
    printf("DSMEM dependent loads: %llu cy / %d loads = %.2f cy/load\n", h, N_LOADS, cy_per_dsmem);

    // Local
    local_cy_test<<<1, 32, 4096>>>(d_out, d_src, d_dst);
    cudaDeviceSynchronize();
    cudaMemcpy(&h, d_out, 8, cudaMemcpyDeviceToHost);
    double cy_per_local = (double)h / N_LOADS;
    printf("Local SMEM dep loads:  %llu cy / %d loads = %.2f cy/load\n", h, N_LOADS, cy_per_local);

    printf("DSMEM/local latency ratio: %.2fx\n", cy_per_dsmem / cy_per_local);

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_out);
    return 0;
}
