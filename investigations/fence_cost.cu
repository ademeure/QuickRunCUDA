// __threadfence variants cost on B300
#include <cuda_runtime.h>
#include <cstdio>

__global__ void no_fence(unsigned *flag, unsigned long long *out, int iters) {
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        flag[blockIdx.x] = i;
        // no fence
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}

__global__ void fence_block(unsigned *flag, unsigned long long *out, int iters) {
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        flag[blockIdx.x] = i;
        __threadfence_block();
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}

__global__ void fence_device(unsigned *flag, unsigned long long *out, int iters) {
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        flag[blockIdx.x] = i;
        __threadfence();
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}

__global__ void fence_system(unsigned *flag, unsigned long long *out, int iters) {
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        flag[blockIdx.x] = i;
        __threadfence_system();
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}

int main() {
    cudaSetDevice(0);
    unsigned *d_flag; cudaMalloc(&d_flag, 1024*sizeof(unsigned));
    unsigned long long *d_out; cudaMalloc(&d_out, 1024*sizeof(unsigned long long));

    int iters = 1000;
    int threads = 32;

    auto run = [&](auto fn, const char *name) {
        // 1 block to measure single-thread cost
        fn<<<1, threads>>>(d_flag, d_out, iters);
        cudaDeviceSynchronize();
        unsigned long long cyc; cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  %-30s %.1f cyc = %.2f ns\n", name, per, per/2.032);
    };

    printf("# B300 fence costs (per fence, in clock64 cycles)\n");
    printf("# 1 block × 32 threads, 1000 store+fence iters\n\n");

    run(no_fence,      "store only (no fence)");
    run(fence_block,   "store + __threadfence_block()");
    run(fence_device,  "store + __threadfence()");
    run(fence_system,  "store + __threadfence_system()");

    return 0;
}
