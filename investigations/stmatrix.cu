// stmatrix variants - inverse of ldmatrix
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>

__global__ void stm_x4(uint32_t *out, int iters, int seed) {
    __shared__ uint32_t smem[2048];
    int tid = threadIdx.x;

    uint32_t r0 = tid + seed, r1 = tid * 2 + seed, r2 = tid * 3 + seed, r3 = tid * 4 + seed;
    uint32_t base = __cvta_generic_to_shared(&smem[tid * 4]);

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        uint32_t addr = base + (i & 0x10);
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared::cta.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
        r0 += 1; r1 += 1; r2 += 1; r3 += 1;
    }
    __syncthreads();
    if (smem[tid] == 0xdeadbeef) out[blockIdx.x] = smem[tid];
}

__global__ void stm_x2(uint32_t *out, int iters, int seed) {
    __shared__ uint32_t smem[1024];
    int tid = threadIdx.x;
    uint32_t r0 = tid + seed, r1 = tid * 2 + seed;
    uint32_t base = __cvta_generic_to_shared(&smem[tid * 2]);

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        uint32_t addr = base + (i & 0x10);
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared::cta.b16 [%0], {%1, %2};\n"
            :: "r"(addr), "r"(r0), "r"(r1));
        r0 += 1; r1 += 1;
    }
    __syncthreads();
    if (smem[tid] == 0xdeadbeef) out[blockIdx.x] = smem[tid];
}

__global__ void stm_x1(uint32_t *out, int iters, int seed) {
    __shared__ uint32_t smem[1024];
    int tid = threadIdx.x;
    uint32_t r0 = tid + seed;
    uint32_t base = __cvta_generic_to_shared(&smem[tid]);

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        uint32_t addr = base + (i & 0x10);
        asm volatile("stmatrix.sync.aligned.x1.m8n8.shared::cta.b16 [%0], {%1};\n"
            :: "r"(addr), "r"(r0));
        r0 += 1;
    }
    __syncthreads();
    if (smem[tid] == 0xdeadbeef) out[blockIdx.x] = smem[tid];
}

int main() {
    cudaSetDevice(0);
    uint32_t *d_out; cudaMalloc(&d_out, 4096 * sizeof(uint32_t));

    int blocks = 148, threads = 32;  // 1 warp per block to start
    int iters = 1000000;

    auto bench = [&](auto launch, int bytes_per_thr) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 3; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            launch();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            if (ms < best) best = ms;
        }
        long total_bytes = (long)blocks * threads * iters * bytes_per_thr;
        double tb = total_bytes / (best/1000.0) / 1e12;
        return std::pair<float, double>{best, tb};
    };

    printf("# B300 stmatrix throughput (148 blocks × 32 thr × %d iter)\n\n", iters);
    printf("# %-25s %-12s %-12s\n", "op", "time_ms", "BW_TB/s");

    auto [t1, bw1] = bench([&]{ stm_x1<<<blocks, threads>>>(d_out, iters, 1); }, 4);
    printf("  %-25s %.2f       %.1f\n", "stmatrix.x1 (4B/thr)", t1, bw1);

    auto [t2, bw2] = bench([&]{ stm_x2<<<blocks, threads>>>(d_out, iters, 1); }, 8);
    printf("  %-25s %.2f       %.1f\n", "stmatrix.x2 (8B/thr)", t2, bw2);

    auto [t4, bw4] = bench([&]{ stm_x4<<<blocks, threads>>>(d_out, iters, 1); }, 16);
    printf("  %-25s %.2f       %.1f\n", "stmatrix.x4 (16B/thr)", t4, bw4);

    return 0;
}
