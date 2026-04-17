// Properly saturate mma.sync to find legacy tensor core peak on B300
// Use 8 ILP, 8 warps per partition (= max occupancy)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define ITERS 5000
#define ILP 8

extern "C" __global__ void k_mma_peak(float *out) {
    unsigned int a0=0x3F803F80,a1=0x3F803F80,a2=0x3F803F80,a3=0x3F803F80;
    unsigned int b0=0x3F803F80,b1=0x3F803F80;
    float d[ILP][4] = {{0}};

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(d[j][0]), "+f"(d[j][1]), "+f"(d[j][2]), "+f"(d[j][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }

    if (threadIdx.x == 0) {
        float s = 0;
        for (int j = 0; j < ILP; j++) for (int k = 0; k < 4; k++) s += d[j][k];
        out[blockIdx.x] = s;
    }
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);

    int sm = prop.multiProcessorCount;
    // Want 4 partitions per SM, 8+ warps per partition. 8 warps × 4 partitions = 32 warps/SM.
    // With block_size 512 (16 warps), 2 blocks per SM = 32 warps total
    int threads = 512;
    int blocks_per_sm = 2;
    int blocks = sm * blocks_per_sm;

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

    float t = bench([&]{ k_mma_peak<<<blocks, threads, 0, s>>>(d_out); });
    long long warps = (long long)blocks * (threads/32);
    long long ops = warps * ITERS * ILP * 4096;  // m16n8k16 = 4096 ops per mma per warp
    printf("# B300 BF16 mma.sync m16n8k16 PEAK (saturated)\n");
    printf("# %d blocks × %d thr (%d warps/SM × %d SMs), %d iter × %d ILP\n",
           blocks, threads, threads/32 * blocks_per_sm, sm, ITERS, ILP);
    printf("# Total mma instructions: %lld\n", warps * ITERS * ILP);
    printf("# Time: %.4f ms\n", t);
    printf("# Throughput: %.2f TFLOPS (BF16→FP32)\n", ops/(t/1e3)/1e12);
    printf("# Spec'd B300 BF16 dense: ~1980 TFLOPS\n");

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
