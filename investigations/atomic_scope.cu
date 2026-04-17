// Atomic scope cost: _block, default (device), _system
#include <cuda_runtime.h>
#include <cstdio>

__global__ void atom_global(unsigned *target, unsigned long long *out, int iters) {
    unsigned a = 0;
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        a += atomicAdd(target, 1);
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
    if (a == 0xdeadbeef) target[1] = a;
}

__global__ void atom_block(unsigned *target, unsigned long long *out, int iters) {
    unsigned a = 0;
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        a += atomicAdd_block(target, 1);
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
    if (a == 0xdeadbeef) target[1] = a;
}

__global__ void atom_system(unsigned *target, unsigned long long *out, int iters) {
    unsigned a = 0;
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        a += atomicAdd_system(target, 1);
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
    if (a == 0xdeadbeef) target[1] = a;
}

__global__ void atom_shared(unsigned *target, unsigned long long *out, int iters) {
    __shared__ unsigned smem;
    if (threadIdx.x == 0) smem = 0;
    __syncthreads();
    unsigned a = 0;
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        a += atomicAdd_block(&smem, 1);
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
    if (a == 0xdeadbeef) target[1] = a;
}

int main() {
    cudaSetDevice(0);
    unsigned *d_target; cudaMalloc(&d_target, 16*sizeof(unsigned));
    unsigned long long *d_out; cudaMalloc(&d_out, 16*sizeof(unsigned long long));

    int iters = 1000;

    auto run = [&](auto fn, int threads, int blocks, const char *name) {
        cudaMemset(d_target, 0, 16*sizeof(unsigned));
        fn<<<blocks, threads>>>(d_target, d_out, iters);
        cudaDeviceSynchronize();
        unsigned long long cyc; cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  %-35s %.1f cyc = %.2f ns\n", name, per, per/2.032);
    };

    printf("# B300 atomic operation cost by scope\n");
    printf("# Single-thread, hot location (no contention)\n\n");
    run(atom_block,  1, 1, "atomicAdd_block (1 thr global)");
    run(atom_shared, 1, 1, "atomicAdd_block (1 thr shared)");
    run(atom_global, 1, 1, "atomicAdd  (1 thr global)");
    run(atom_system, 1, 1, "atomicAdd_system (1 thr global)");

    printf("\n# 32-thread warp, all hitting same location:\n");
    run(atom_block,  32, 1, "atomicAdd_block (32 thr global)");
    run(atom_shared, 32, 1, "atomicAdd_block (32 thr shared)");
    run(atom_global, 32, 1, "atomicAdd  (32 thr global)");
    run(atom_system, 32, 1, "atomicAdd_system (32 thr global)");

    return 0;
}
