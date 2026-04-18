// Beat cudaMemset with optimized custom kernel
#include <cuda_runtime.h>
#include <cstdio>

// Variants of write kernels with different thread counts and vector widths

// V1: 8-ILP × 16-byte (int4) stores, 256 thr/block × 8 blocks/SM
__launch_bounds__(256, 8) __global__ void w_int4_256(int4 *data, int N, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int4 v = make_int4(seed, seed+1, seed+2, seed+3);
    for (int i = tid; i < N - 7*stride; i += 8*stride) {
        data[i] = v; data[i+stride] = v; data[i+2*stride] = v; data[i+3*stride] = v;
        data[i+4*stride] = v; data[i+5*stride] = v; data[i+6*stride] = v; data[i+7*stride] = v;
    }
}

// V2: same but 1024 threads × 2 blocks/SM (full occupancy)
__launch_bounds__(1024, 2) __global__ void w_int4_1024(int4 *data, int N, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int4 v = make_int4(seed, seed+1, seed+2, seed+3);
    for (int i = tid; i < N - 7*stride; i += 8*stride) {
        data[i] = v; data[i+stride] = v; data[i+2*stride] = v; data[i+3*stride] = v;
        data[i+4*stride] = v; data[i+5*stride] = v; data[i+6*stride] = v; data[i+7*stride] = v;
    }
}

// V3: 512 threads × 4 blocks/SM
__launch_bounds__(512, 4) __global__ void w_int4_512(int4 *data, int N, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int4 v = make_int4(seed, seed+1, seed+2, seed+3);
    for (int i = tid; i < N - 7*stride; i += 8*stride) {
        data[i] = v; data[i+stride] = v; data[i+2*stride] = v; data[i+3*stride] = v;
        data[i+4*stride] = v; data[i+5*stride] = v; data[i+6*stride] = v; data[i+7*stride] = v;
    }
}

// V4: 16-ILP unrolled, 256 thr × 8 blk/SM
__launch_bounds__(256, 8) __global__ void w_int4_256_ilp16(int4 *data, int N, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int4 v = make_int4(seed, seed+1, seed+2, seed+3);
    for (int i = tid; i < N - 15*stride; i += 16*stride) {
        #pragma unroll
        for (int j = 0; j < 16; j++) data[i + j*stride] = v;
    }
}

int main() {
    cudaSetDevice(0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    size_t bytes = 4096ul * 1024 * 1024;
    int N = bytes / 16;
    int4 *d; cudaMalloc(&d, bytes);

    auto bench = [&](auto launch) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 7; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;
    };

    auto report = [&](float t, const char *name) {
        double bw = bytes/(t/1000)/1e9;
        printf("  %-30s %.3f ms  %.0f GB/s  %.1f%%\n", name, t, bw, bw/7672*100);
    };

    printf("# Beat cudaMemset: 4 GB writes, various configs\n\n");
    printf("# %-30s %-12s %-12s %-12s\n", "kernel", "ms", "GB/s", "%peak");

    report(bench([&]{ cudaMemsetAsync(d, 0xab, bytes, 0); }), "cudaMemset (reference)");
    report(bench([&]{ w_int4_256<<<148, 256>>>(d, N, 1); }), "256 thr × 148 blk, 8-ILP");
    report(bench([&]{ w_int4_512<<<148, 512>>>(d, N, 1); }), "512 thr × 148 blk, 8-ILP");
    report(bench([&]{ w_int4_1024<<<296, 1024>>>(d, N, 1); }), "1024 thr × 296 blk, 8-ILP");
    report(bench([&]{ w_int4_256<<<1184, 256>>>(d, N, 1); }), "256 thr × 1184 blk (8/SM), 8-ILP");
    report(bench([&]{ w_int4_256_ilp16<<<1184, 256>>>(d, N, 1); }), "256 thr × 1184 blk, 16-ILP");
    report(bench([&]{ w_int4_512<<<592, 512>>>(d, N, 1); }), "512 thr × 592 blk (4/SM), 8-ILP");

    return 0;
}
