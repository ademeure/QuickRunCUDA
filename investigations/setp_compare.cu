// PTX setp comparison cost variants
#include <cuda_runtime.h>
#include <cstdio>

__global__ void cmp_int(int *out, int iters, int k) {
    int a = threadIdx.x;
    int sum = 0;
    for (int i = 0; i < iters; i++) {
        if (a < k) sum += a;
        else sum -= a;
        a = a * 7 + 1;
    }
    if (sum == 0xdeadbeef) out[blockIdx.x] = sum;
}

__global__ void cmp_uint(unsigned *out, int iters, unsigned k) {
    unsigned a = threadIdx.x + 1;
    unsigned sum = 0;
    for (int i = 0; i < iters; i++) {
        if (a < k) sum += a;
        else sum -= a;
        a = a * 7 + 1;
    }
    if (sum == 0xdeadbeef) out[blockIdx.x] = sum;
}

__global__ void cmp_float(float *out, int iters, float k) {
    float a = threadIdx.x;
    float sum = 0;
    for (int i = 0; i < iters; i++) {
        if (a < k) sum += a;
        else sum -= a;
        a = a * 1.0001f + 0.0001f;
    }
    if (sum < -1e30f) out[blockIdx.x] = sum;
}

__global__ void cmp_long(long long *out, int iters, long long k) {
    long long a = threadIdx.x;
    long long sum = 0;
    for (int i = 0; i < iters; i++) {
        if (a < k) sum += a;
        else sum -= a;
        a = a * 7 + 1;
    }
    if (sum == 0xdeadbeef) out[blockIdx.x] = sum;
}

int main() {
    cudaSetDevice(0);
    void *d_out; cudaMalloc(&d_out, 1024 * sizeof(long long));
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
        printf("  %-30s %.3f ms\n", name, best);
    };

    printf("# B300 setp (compare) variants\n\n");
    bench([&]{ cmp_int<<<blocks, threads>>>((int*)d_out, iters, 1000); }, "setp.lt.s32");
    bench([&]{ cmp_uint<<<blocks, threads>>>((unsigned*)d_out, iters, 1000); }, "setp.lt.u32");
    bench([&]{ cmp_float<<<blocks, threads>>>((float*)d_out, iters, 100.0f); }, "setp.lt.f32");
    bench([&]{ cmp_long<<<blocks, threads>>>((long long*)d_out, iters, 1000); }, "setp.lt.s64");

    return 0;
}
