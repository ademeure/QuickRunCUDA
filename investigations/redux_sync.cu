// Direct PTX redux.sync ops
#include <cuda_runtime.h>
#include <cstdio>

__global__ void redux_add(unsigned *out, int iters) {
    unsigned a = threadIdx.x + 1;
    for (int i = 0; i < iters; i++) {
        unsigned r;
        asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(a));
        a = r + i;
    }
    if (a == 0xdeadbeef) out[blockIdx.x] = a;
}

__global__ void redux_min(unsigned *out, int iters) {
    unsigned a = threadIdx.x + 1;
    for (int i = 0; i < iters; i++) {
        unsigned r;
        asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(a));
        a = r + i;
    }
    if (a == 0xdeadbeef) out[blockIdx.x] = a;
}

__global__ void redux_max(unsigned *out, int iters) {
    unsigned a = threadIdx.x + 1;
    for (int i = 0; i < iters; i++) {
        unsigned r;
        asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(a));
        a = r + i;
    }
    if (a == 0xdeadbeef) out[blockIdx.x] = a;
}

__global__ void redux_or(unsigned *out, int iters) {
    unsigned a = threadIdx.x + 1;
    for (int i = 0; i < iters; i++) {
        unsigned r;
        asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(a));
        a = r ^ i;  // mix in i
    }
    if (a == 0xdeadbeef) out[blockIdx.x] = a;
}

int main() {
    cudaSetDevice(0);
    unsigned *d_out; cudaMalloc(&d_out, 1024 * sizeof(unsigned));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 128;

    auto bench = [&](auto launch, const char *name) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 3; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long total = (long)blocks * threads * iters;
        double gops = total / (best/1000.0) / 1e9;
        printf("  %-25s %.3f ms = %.0f Gops/s\n", name, best, gops);
    };

    printf("# B300 redux.sync direct PTX (148 × 128 thr × 100k iter)\n\n");
    bench([&]{ redux_add<<<blocks, threads>>>(d_out, iters); }, "redux.sync.add.u32");
    bench([&]{ redux_min<<<blocks, threads>>>(d_out, iters); }, "redux.sync.min.u32");
    bench([&]{ redux_max<<<blocks, threads>>>(d_out, iters); }, "redux.sync.max.u32");
    bench([&]{ redux_or<<<blocks, threads>>>(d_out, iters); }, "redux.sync.or.b32");

    return 0;
}
