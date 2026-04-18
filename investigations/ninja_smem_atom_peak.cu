// SMEM atomic peak throughput on B300
//
// Theoretical:
//   SMEM has 32 banks/SM, 1 atomic op/bank/cy
//   Per SM: 32 ops/cy  (when no bank conflicts)
//   Aggregate: 32 * 148 * 2.032e9 = 9.6 Tops/s atomic adds peak
//
// Method: each thread does atomicAdd_block to SMEM at strided index
//   to avoid bank conflicts. Vary stride and ILP.
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void smem_atom_strided(int *out, int N_iters, int stride) {
    extern __shared__ int smem[];  // 32 ints initialized to 0
    int tid = threadIdx.x;
    int bank = tid & 31;
    int *target = &smem[bank * stride];
    if (tid < 32 * stride) smem[tid] = 0;
    __syncthreads();

    for (int i = 0; i < N_iters; i++) {
        atomicAdd_block(target, 1);
    }
    __syncthreads();
    if (tid == 0) out[blockIdx.x] = smem[0];
}

template <int ILP>
__launch_bounds__(256, 8) __global__ void smem_atom_ilp(int *out, int N_iters) {
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    int targets[ILP];
    #pragma unroll
    for (int j = 0; j < ILP; j++) targets[j] = (tid * ILP + j) & 31;  // each thread has ILP targets, all distinct banks?
    if (tid < 32) smem[tid] = 0;
    __syncthreads();

    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            atomicAdd_block(&smem[targets[j]], 1);
        }
    }
    __syncthreads();
    if (tid == 0) out[blockIdx.x] = smem[0];
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int blocks = 148 * 8;  // 8 blocks/SM, full occupancy
    int N = 100000;
    int smem_bytes = 32 * 64 * sizeof(int);  // 32 banks * stride 64

    auto run = [&](const char* name, void(*kfn)(int*, int, int), int stride, int N_iters_local) {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        for (int i = 0; i < 3; i++) kfn<<<blocks, 256, smem_bytes>>>(d_out, N_iters_local, stride);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s: %s\n", name, cudaGetErrorString(cudaGetLastError())); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, 256, smem_bytes>>>(d_out, N_iters_local, stride);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long total = (long)blocks * 256 * N_iters_local;
        double gops = total / (best/1000.0) / 1e9;
        printf("  %-25s stride=%d  %.4f ms = %.2f Gops/s atomic\n", name, stride, best, gops);
    };

    printf("# SMEM atomic strided sweep (256 thr/block, %d blocks)\n", blocks);
    run("strided", smem_atom_strided, 1, N);
    run("strided", smem_atom_strided, 2, N);
    run("strided", smem_atom_strided, 4, N);
    run("strided", smem_atom_strided, 8, N);

    return 0;
}
