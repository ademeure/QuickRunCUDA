// Tensor core MMA with ILP=4 (4 independent accumulator chains)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define ITERS 5000

extern "C" __global__ void k_mma_ilp4(float *out) {
    unsigned int a0 = 0x3F803F80, a1 = 0x3F803F80, a2 = 0x3F803F80, a3 = 0x3F803F80;
    unsigned int b0 = 0x3F803F80, b1 = 0x3F803F80;
    // 4 independent accumulator sets
    float d0[4] = {0,0,0,0}, d1[4] = {0,0,0,0}, d2[4] = {0,0,0,0}, d3[4] = {0,0,0,0};

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d1[0]), "+f"(d1[1]), "+f"(d1[2]), "+f"(d1[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d2[0]), "+f"(d2[1]), "+f"(d2[2]), "+f"(d2[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d3[0]), "+f"(d3[1]), "+f"(d3[2]), "+f"(d3[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    if (threadIdx.x == 0) {
        float s = 0;
        for (int i = 0; i < 4; i++) s += d0[i] + d1[i] + d2[i] + d3[i];
        out[blockIdx.x] = s;
    }
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

    float t = bench([&]{ k_mma_ilp4<<<blocks, threads, 0, s>>>(d_out); });
    long long warps = (long long)blocks * (threads/32);
    long long ops = warps * ITERS * 4096 * 4;  // 4 mma per iter, each 4096 ops
    printf("# B300 BF16 Tensor Core: m16n8k16 with ILP=4\n");
    printf("# %d blocks × %d thr × %d iter × 4 mma = %lld total mma\n",
           blocks, threads, ITERS, warps * ITERS * 4);
    printf("# Time: %.4f ms\n", t);
    printf("# Throughput: %.2f TFLOPS\n", ops/(t/1e3)/1e12);

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
