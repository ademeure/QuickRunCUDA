// __syncwarp cost (Volta+ explicit warp sync)
#include <cuda_runtime.h>
#include <cstdio>

__global__ void sync_warp_loop(unsigned long long *out, int iters) {
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        __syncwarp();
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}

__global__ void no_sync_loop(unsigned long long *out, int iters) {
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        // empty
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}

__global__ void shfl_sync_loop(unsigned long long *out, int iters) {
    int v = threadIdx.x;
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        v = __shfl_sync(0xffffffff, v, 0);
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0 + v;
}

__global__ void bar_sync_loop(unsigned long long *out, int iters) {
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        asm volatile("bar.warp.sync 0xffffffff;");
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out; cudaMalloc(&d_out, 16 * sizeof(unsigned long long));

    int iters = 1000;

    auto run = [&](auto launch, const char *name) {
        launch();
        cudaDeviceSynchronize();
        unsigned long long cyc; cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  %-30s %.1f cyc = %.2f ns\n", name, per, per/2.032);
    };

    printf("# B300 warp sync cost (single warp loop)\n\n");

    run([&]{ no_sync_loop<<<1, 32>>>(d_out, iters); }, "empty loop");
    run([&]{ sync_warp_loop<<<1, 32>>>(d_out, iters); }, "__syncwarp()");
    run([&]{ bar_sync_loop<<<1, 32>>>(d_out, iters); }, "bar.warp.sync (PTX)");
    run([&]{ shfl_sync_loop<<<1, 32>>>(d_out, iters); }, "shfl_sync (broadcast)");

    return 0;
}
