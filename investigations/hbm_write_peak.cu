// HBM write peak with high ILP
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void __launch_bounds__(512, 4) write_ilp(int4 *data, int N, int seed, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int4 v0 = make_int4(seed, seed+1, seed+2, seed+3);
    int4 v1 = make_int4(seed*2, seed*2+1, seed*2+2, seed*2+3);
    int4 v2 = make_int4(seed*3, seed*3+1, seed*3+2, seed*3+3);
    int4 v3 = make_int4(seed*4, seed*4+1, seed*4+2, seed*4+3);
    int4 v4 = make_int4(seed*5, seed*5+1, seed*5+2, seed*5+3);
    int4 v5 = make_int4(seed*6, seed*6+1, seed*6+2, seed*6+3);
    int4 v6 = make_int4(seed*7, seed*7+1, seed*7+2, seed*7+3);
    int4 v7 = make_int4(seed*8, seed*8+1, seed*8+2, seed*8+3);

    for (int it = 0; it < iters; it++) {
        for (int i = tid; i < N - 7*stride; i += 8 * stride) {
            data[i] = v0;
            data[i + stride] = v1;
            data[i + 2*stride] = v2;
            data[i + 3*stride] = v3;
            data[i + 4*stride] = v4;
            data[i + 5*stride] = v5;
            data[i + 6*stride] = v6;
            data[i + 7*stride] = v7;
        }
    }
}

int main() {
    cudaSetDevice(0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    size_t bytes = 4096ul * 1024 * 1024;  // 4 GB
    int N = bytes / 16;

    int4 *d_data; cudaMalloc(&d_data, bytes);

    int iters = 3;
    int blocks = 148, threads = 512;

    for (int i = 0; i < 3; i++) write_ilp<<<blocks, threads>>>(d_data, N, 1, 1);
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int trial = 0; trial < 5; trial++) {
        cudaEventRecord(e0);
        write_ilp<<<blocks, threads>>>(d_data, N, 1, iters);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }

    double total = (double)bytes * iters;
    double bw = total / (best/1000.0) / 1e9;
    printf("# B300 HBM write peak (8-way ILP, 512 thr/blk, 4 GB)\n");
    printf("  Time: %.2f ms\n", best);
    printf("  BW:   %.0f GB/s (%.0f%% of HBM peak 7672)\n", bw, bw/7672*100);

    return 0;
}
