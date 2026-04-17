// Integer arithmetic throughput on B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define ITERS 50000
#define ILP 8

extern "C" __global__ void int_add(int *in, int *out) {
    int a[ILP];
    for (int i = 0; i < ILP; i++) a[i] = in[i];
    int b = in[8], c = in[9];

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) a[j] = a[j] + b + c;
    }

    if (threadIdx.x == 0) {
        int s = 0;
        for (int i = 0; i < ILP; i++) s += a[i];
        out[blockIdx.x] = s;
    }
}

extern "C" __global__ void int_mad(int *in, int *out) {
    int a[ILP];
    for (int i = 0; i < ILP; i++) a[i] = in[i];
    int b = in[8], c = in[9];

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++)
            asm volatile("mad.lo.s32 %0, %0, %1, %2;" : "+r"(a[j]) : "r"(b), "r"(c));
    }

    if (threadIdx.x == 0) {
        int s = 0;
        for (int i = 0; i < ILP; i++) s += a[i];
        out[blockIdx.x] = s;
    }
}

extern "C" __global__ void int_mul(int *in, int *out) {
    int a[ILP];
    for (int i = 0; i < ILP; i++) a[i] = in[i];
    int b = in[8];

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) a[j] = a[j] * b;
    }

    if (threadIdx.x == 0) {
        int s = 0;
        for (int i = 0; i < ILP; i++) s += a[i];
        out[blockIdx.x] = s;
    }
}

extern "C" __global__ void int_shl(int *in, int *out) {
    int a[ILP];
    for (int i = 0; i < ILP; i++) a[i] = in[i];

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) a[j] = a[j] << 3;
    }

    if (threadIdx.x == 0) {
        int s = 0;
        for (int i = 0; i < ILP; i++) s += a[i];
        out[blockIdx.x] = s;
    }
}

extern "C" __global__ void int_xor(int *in, int *out) {
    int a[ILP];
    for (int i = 0; i < ILP; i++) a[i] = in[i];
    int b = in[8];

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) a[j] = a[j] ^ b;
    }

    if (threadIdx.x == 0) {
        int s = 0;
        for (int i = 0; i < ILP; i++) s += a[i];
        out[blockIdx.x] = s;
    }
}

extern "C" __global__ void int_popc(int *in, int *out) {
    int a[ILP];
    for (int i = 0; i < ILP; i++) a[i] = in[i];

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) a[j] = __popc(a[j] * 31 + j);
    }

    if (threadIdx.x == 0) {
        int s = 0;
        for (int i = 0; i < ILP; i++) s += a[i];
        out[blockIdx.x] = s;
    }
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm = prop.multiProcessorCount;

    int *d_in, *d_out;
    cudaMalloc(&d_in, 16 * sizeof(int));
    cudaMalloc(&d_out, sm * sizeof(int));

    int h_in[16];
    for (int i = 0; i < 16; i++) h_in[i] = 1 + i * 7;
    cudaMemcpy(d_in, h_in, 16 * sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 2; i++) { fn(); cudaDeviceSynchronize(); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    auto run = [&](const char *name, void (*fn_ptr)(int*, int*)) {
        float t = bench([&]{ fn_ptr<<<sm, 128, 0, s>>>(d_in, d_out); });
        long long ops = (long long)sm * 128 * ITERS * ILP;
        double tops = (double)ops / (t/1e3) / 1e12;
        printf("  %-15s : %.3f ms, %.2f Gops/s/SM\n", name, t, tops*1e3/sm);
    };

    printf("# B300 integer arithmetic throughput\n");
    printf("# 148 blocks × 128 threads × ILP=%d × %d iters\n\n", ILP, ITERS);

    run("int add",    int_add);
    run("int mad.lo", int_mad);
    run("int mul",    int_mul);
    run("int shl",    int_shl);
    run("int xor",    int_xor);
    run("int __popc", int_popc);

    cudaStreamDestroy(s);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
