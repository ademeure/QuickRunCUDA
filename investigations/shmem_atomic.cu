// SHMEM atomic throughput on B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define ITERS 10000

extern "C" __global__ void shmem_atomic_uncontend(unsigned long long *out) {
    __shared__ unsigned int counters[1024];
    int tid = threadIdx.x;
    counters[tid] = 0;
    __syncthreads();

    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    unsigned int v;
    for (int i = 0; i < ITERS; i++) {
        // Inline PTX so compiler can't elide
        asm volatile("atom.shared.add.u32 %0, [%1], %2;"
                     : "=r"(v) : "l"(&counters[tid]), "r"(1));
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[0] = end - start;
        out[8] = v;  // defeat DCE
    }
}

extern "C" __global__ void shmem_atomic_warp_contend(unsigned long long *out) {
    __shared__ unsigned int counter;
    int tid = threadIdx.x;
    if (tid == 0) counter = 0;
    __syncthreads();

    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    unsigned int v;
    for (int i = 0; i < ITERS; i++) {
        asm volatile("atom.shared.add.u32 %0, [%1], %2;"
                     : "=r"(v) : "l"(&counter), "r"(1));
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[0] = end - start;
        out[8] = v;
    }
}

extern "C" __global__ void shmem_atomic_cas(unsigned long long *out) {
    __shared__ unsigned int counters[1024];
    int tid = threadIdx.x;
    counters[tid] = 0;
    __syncthreads();

    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    for (int i = 0; i < ITERS; i++) {
        atomicCAS(&counters[tid], 0, 1);  // always fails second time
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[0] = end - start;
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out;
    cudaMalloc(&d_out, 8 * sizeof(unsigned long long));

    // Warmup
    shmem_atomic_uncontend<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();

    shmem_atomic_uncontend<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    unsigned long long h_un;
    cudaMemcpy(&h_un, d_out, 8, cudaMemcpyDeviceToHost);

    shmem_atomic_warp_contend<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    unsigned long long h_cn;
    cudaMemcpy(&h_cn, d_out, 8, cudaMemcpyDeviceToHost);

    shmem_atomic_cas<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    unsigned long long h_cas;
    cudaMemcpy(&h_cas, d_out, 8, cudaMemcpyDeviceToHost);

    printf("# B300 SHMEM atomic cost (single warp)\n");
    printf("# %d iters per config\n\n", ITERS);
    printf("  uncontended atomicAdd (per-thread addr): %llu cy total = %.2f cy/iter\n",
           h_un, h_un / (double)ITERS);
    printf("  warp-contended atomicAdd (all same addr): %llu cy = %.2f cy/iter\n",
           h_cn, h_cn / (double)ITERS);
    printf("  uncontended atomicCAS (per-thread addr):  %llu cy = %.2f cy/iter\n",
           h_cas, h_cas / (double)ITERS);

    cudaFree(d_out);
    return 0;
}
