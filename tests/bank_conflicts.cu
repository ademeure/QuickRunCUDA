// Shared memory bank conflict measurement
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define ITERS 1000

extern "C" __global__ void no_conflict(unsigned long long *out) {
    __shared__ float buf[1024];
    int tid = threadIdx.x;
    if (tid < 1024) buf[tid] = (float)tid;
    __syncthreads();

    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    float a = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Different address each iter — prevent hoisting
        a += buf[(tid + i) & 31];
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0 && a < 1e30f) out[blockIdx.x] = end - start;
}

extern "C" __global__ void conflict_2way(unsigned long long *out) {
    __shared__ float buf[1024];
    int tid = threadIdx.x;
    if (tid < 1024) buf[tid] = (float)tid;
    __syncthreads();

    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    float a = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // 2-way: tid 0+16 hit bank 0+16, tid 1+17 hit bank 1+17, ...
        // Actually do this: tid 0..15 get banks 0..15, tid 16..31 ALSO get banks 0..15 (with offset 32)
        unsigned int idx = ((tid & 15) | ((i & 1) << 5));
        a += buf[idx];
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0 && a < 1e30f) out[blockIdx.x] = end - start;
}

extern "C" __global__ void conflict_32way(unsigned long long *out) {
    __shared__ float buf[1024];
    int tid = threadIdx.x;
    if (tid < 1024) buf[tid] = (float)tid;
    __syncthreads();

    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    float a = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // tid * 32 = bank 0 for all, but addresses 0,32,64,...,992 = 32 distinct words → 32-way conflict
        unsigned int idx = (tid * 32 + (i & 1)) & 1023;
        a += buf[idx];
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0 && a < 1e30f) out[blockIdx.x] = end - start;
}

extern "C" __global__ void broadcast(unsigned long long *out) {
    __shared__ float buf[1024];
    int tid = threadIdx.x;
    if (tid < 1024) buf[tid] = (float)tid;
    __syncthreads();

    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    float a = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // All threads read SAME address (broadcast - free)
        a += buf[i & 1023];
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0 && a < 1e30f) out[blockIdx.x] = end - start;
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out;
    cudaMalloc(&d_out, 16);

    auto run_test = [&](const char *name, void (*fn)(unsigned long long*)) {
        fn<<<1, 32>>>(d_out);
        cudaDeviceSynchronize();
        unsigned long long h;
        cudaMemcpy(&h, d_out, 8, cudaMemcpyDeviceToHost);
        printf("  %-20s : %llu cycles for %d iters = %.2f cy/iter\n",
               name, h, ITERS, (double)h / ITERS);
    };

    printf("# B300 shmem bank conflict measurement (1 warp = 32 threads)\n\n");
    run_test("No conflict", no_conflict);
    run_test("2-way conflict", conflict_2way);
    run_test("32-way conflict", conflict_32way);
    run_test("Broadcast (all same)", broadcast);

    cudaFree(d_out);
    return 0;
}
