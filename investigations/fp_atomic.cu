// Float atomic operations cost
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

__global__ void f32_atomic(float *target, unsigned long long *out, int iters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        for (int i = 0; i < iters; i++) {
            atomicAdd(target, 1.0f);
        }
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    }
}

__global__ void f16_atomic(__half *target, unsigned long long *out, int iters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        for (int i = 0; i < iters; i++) {
            atomicAdd(target, __float2half(1.0f));
        }
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    }
}

__global__ void bf16_atomic(__nv_bfloat16 *target, unsigned long long *out, int iters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        for (int i = 0; i < iters; i++) {
            atomicAdd(target, __float2bfloat16(1.0f));
        }
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    }
}

__global__ void f64_atomic(double *target, unsigned long long *out, int iters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        for (int i = 0; i < iters; i++) {
            atomicAdd(target, 1.0);
        }
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    }
}

int main() {
    cudaSetDevice(0);
    float *d_f32; cudaMalloc(&d_f32, sizeof(float));
    __half *d_f16; cudaMalloc(&d_f16, sizeof(__half));
    __nv_bfloat16 *d_bf16; cudaMalloc(&d_bf16, sizeof(__nv_bfloat16));
    double *d_f64; cudaMalloc(&d_f64, sizeof(double));
    unsigned long long *d_out; cudaMalloc(&d_out, sizeof(unsigned long long));

    int iters = 1000;
    cudaMemset(d_f32, 0, sizeof(float));
    cudaMemset(d_f16, 0, sizeof(__half));
    cudaMemset(d_bf16, 0, sizeof(__nv_bfloat16));
    cudaMemset(d_f64, 0, sizeof(double));

    auto run = [&](auto launch, const char *name) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        unsigned long long cyc; cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  %-25s %.1f cyc = %.2f ns\n", name, per, per/2.032);
    };

    printf("# B300 FP atomic operation costs (single thread, 1000 iter chained)\n\n");

    run([&]{ f32_atomic<<<1, 32>>>(d_f32, d_out, iters); }, "atomicAdd float (FP32)");
    run([&]{ f16_atomic<<<1, 32>>>(d_f16, d_out, iters); }, "atomicAdd __half");
    run([&]{ bf16_atomic<<<1, 32>>>(d_bf16, d_out, iters); }, "atomicAdd __nv_bfloat16");
    run([&]{ f64_atomic<<<1, 32>>>(d_f64, d_out, iters); }, "atomicAdd double (FP64)");

    return 0;
}
