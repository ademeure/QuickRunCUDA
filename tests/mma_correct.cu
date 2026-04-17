// Tensor core mma.sync test with CORRECT vector sizes
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define ITERS 5000

// m16n8k8 BF16: A 2×.b32, B 1×.b32, D/C 4×.f32
extern "C" __global__ void k_mma_m16n8k8(float *out) {
    unsigned int a0 = 0x3F803F80, a1 = 0x3F803F80;
    unsigned int b0 = 0x3F803F80;
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(b0)
        );
    }
    if (threadIdx.x == 0) out[blockIdx.x] = d0 + d1 + d2 + d3;
}

// m16n8k16 BF16: A 4×.b32, B 2×.b32, D/C 4×.f32
extern "C" __global__ void k_mma_m16n8k16(float *out) {
    unsigned int a0 = 0x3F803F80, a1 = 0x3F803F80, a2 = 0x3F803F80, a3 = 0x3F803F80;
    unsigned int b0 = 0x3F803F80, b1 = 0x3F803F80;
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
        );
    }
    if (threadIdx.x == 0) out[blockIdx.x] = d0 + d1 + d2 + d3;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount * 4, threads = 256;

    float *d_out;
    cudaMalloc(&d_out, blocks * sizeof(float));

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

    {
        float t = bench([&]{ k_mma_m16n8k8<<<blocks, threads, 0, s>>>(d_out); });
        // Each m16n8k8 BF16 mma = M*N*K*2 = 16*8*8*2 = 2048 ops per warp per iter
        long long warps = (long long)blocks * (threads/32);
        long long ops = warps * ITERS * 2048;
        printf("m16n8k8 BF16: %.4f ms = %.2f TFLOPS\n", t, ops/(t/1e3)/1e12);
    }

    {
        float t = bench([&]{ k_mma_m16n8k16<<<blocks, threads, 0, s>>>(d_out); });
        // m16n8k16 BF16 = 16*8*16*2 = 4096 ops per warp per iter
        long long warps = (long long)blocks * (threads/32);
        long long ops = warps * ITERS * 4096;
        printf("m16n8k16 BF16: %.4f ms = %.2f TFLOPS\n", t, ops/(t/1e3)/1e12);
    }

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
