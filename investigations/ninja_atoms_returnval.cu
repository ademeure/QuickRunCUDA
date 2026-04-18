// Test ATOMS.POPC.INC with vs without using return value
#include <cuda_runtime.h>
#include <cstdio>

// (1) Discard return - what we tested before
__launch_bounds__(256, 8) __global__ void smem_add_discard(int *out, int N_iters) {
    __shared__ int smem[64];
    if (threadIdx.x < 64) smem[threadIdx.x] = 0;
    __syncthreads();
    for (int i = 0; i < N_iters; i++) {
        atomicAdd_block(&smem[0], 1);  // discard return
    }
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = smem[0];
}

// (2) USE return value (chained as accumulator)
__launch_bounds__(256, 8) __global__ void smem_add_useret(int *out, int N_iters) {
    __shared__ int smem[64];
    if (threadIdx.x < 64) smem[threadIdx.x] = 0;
    __syncthreads();
    int acc = 0;
    for (int i = 0; i < N_iters; i++) {
        int prev = atomicAdd_block(&smem[0], 1);
        acc += prev;
    }
    __syncthreads();
    if (acc == 0xdeadbeef) out[blockIdx.x] = acc;
    else if (threadIdx.x == 0) out[blockIdx.x + 1024] = smem[0];
}

// (3) USE return value where it affects address (forces real dep)
__launch_bounds__(256, 8) __global__ void smem_add_chain(int *out, int N_iters) {
    __shared__ int smem[64];
    if (threadIdx.x < 64) smem[threadIdx.x] = 0;
    __syncthreads();
    int idx = 0;
    for (int i = 0; i < N_iters; i++) {
        idx = atomicAdd_block(&smem[idx & 0], 1) & 0;  // dep on prev result
    }
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = smem[0];
}

int main() {
    cudaSetDevice(0);
    int N_iters = 10000;
    int *d_out; cudaMalloc(&d_out, 4 * 1024 * 1024); cudaMemset(d_out, 0, 4 * 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;

    auto run = [&](const char* name, void(*kfn)(int*, int)) {
        for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_out, N_iters);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", name); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, threads>>>(d_out, N_iters);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * N_iters;
        double T = ops / (best/1000.0) / 1e9;
        double T_per_sm_per_cy = T / 148 / 2.032;
        printf("  %-30s %.3f ms  T=%5.0f Gops  per-SM-cy=%.2f\n",
            name, best, T, T_per_sm_per_cy);
    };

    printf("# ATOMS.POPC.INC return-value semantics test\n\n");
    run("(1) discard return",        smem_add_discard);
    run("(2) use return (acc sum)",  smem_add_useret);
    run("(3) use return (addr dep)", smem_add_chain);
    return 0;
}
