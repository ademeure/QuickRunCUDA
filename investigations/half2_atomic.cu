// Test packed half2/bfloat162 atomic add
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

__global__ void f32_atom(float *t, unsigned long long *out, int iters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        for (int i = 0; i < iters; i++) atomicAdd(t, 1.0f);
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    }
}

__global__ void h2_atom(__half2 *t, unsigned long long *out, int iters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        __half2 v = __float2half2_rn(1.0f);
        for (int i = 0; i < iters; i++) atomicAdd(t, v);
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    }
}

__global__ void bf2_atom(__nv_bfloat162 *t, unsigned long long *out, int iters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        __nv_bfloat162 v;
        v.x = __float2bfloat16(1.0f);
        v.y = __float2bfloat16(1.0f);
        for (int i = 0; i < iters; i++) atomicAdd(t, v);
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    }
}

// Native PTX red.add.noftz.f16 (HW reduction - check if available)
__global__ void f16_red_native(__half *t, unsigned long long *out, int iters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        for (int i = 0; i < iters; i++) {
            asm volatile("red.global.add.noftz.f16 [%0], %1;" :: "l"(t), "h"((unsigned short)0x3c00) : "memory");
        }
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    }
}

int main() {
    cudaSetDevice(0);
    float *d_f32; cudaMalloc(&d_f32, sizeof(float));
    __half2 *d_h2; cudaMalloc(&d_h2, sizeof(__half2));
    __nv_bfloat162 *d_bf2; cudaMalloc(&d_bf2, sizeof(__nv_bfloat162));
    __half *d_h; cudaMalloc(&d_h, sizeof(__half));
    unsigned long long *d_out; cudaMalloc(&d_out, sizeof(unsigned long long));

    int iters = 1000;
    cudaMemset(d_f32, 0, sizeof(float));
    cudaMemset(d_h2, 0, sizeof(__half2));
    cudaMemset(d_bf2, 0, sizeof(__nv_bfloat162));
    cudaMemset(d_h, 0, sizeof(__half));

    auto run = [&](auto launch, const char *name) {
        for (int i = 0; i < 3; i++) launch();
        cudaError_t err = cudaDeviceSynchronize();
        if (err) { printf("  %-30s ERR: %s\n", name, cudaGetErrorString(err)); return; }
        unsigned long long cyc; cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  %-30s %.1f cyc = %.2f ns\n", name, per, per/2.032);
    };

    printf("# B300 packed FP atomic vs scalar\n\n");
    run([&]{ f32_atom<<<1, 32>>>(d_f32, d_out, iters); }, "atomicAdd float (baseline)");
    run([&]{ h2_atom<<<1, 32>>>(d_h2, d_out, iters); }, "atomicAdd half2");
    run([&]{ bf2_atom<<<1, 32>>>(d_bf2, d_out, iters); }, "atomicAdd bfloat162");
    run([&]{ f16_red_native<<<1, 32>>>(d_h, d_out, iters); }, "red.add.noftz.f16 (PTX)");

    return 0;
}
