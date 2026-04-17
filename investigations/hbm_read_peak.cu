// Peak HBM read bandwidth test
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void read_only(const int4 *data, int *out, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int4 acc = make_int4(0, 0, 0, 0);
    for (int it = 0; it < iters; it++) {
        for (int i = tid; i < N; i += stride) {
            int4 v = data[i];
            acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
        }
    }
    if (acc.x == 0xdeadbeef) out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}

int main() {
    cudaSetDevice(0);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("# B300 HBM read bandwidth (with int4 vector loads)\n");
    printf("# Theoretical HBM3E peak: 7672 GB/s\n\n");
    printf("# %-12s %-12s %-12s %-12s\n", "size", "iters", "time_ms", "BW_GB/s");

    int blocks = 148, threads = 256;

    for (size_t mb : {16, 64, 256, 1024, 4096}) {
        size_t bytes = mb * 1024 * 1024;
        int N = bytes / 16;  // int4 = 16 B
        int iters = (mb == 16 || mb == 64) ? 200 : (mb == 256 ? 50 : (mb == 1024 ? 10 : 3));

        int4 *d_data; cudaMalloc(&d_data, bytes);
        cudaMemset(d_data, 0xab, bytes);
        int *d_out; cudaMalloc(&d_out, blocks * threads * sizeof(int));

        // Warmup
        for (int i = 0; i < 3; i++) read_only<<<blocks, threads>>>(d_data, d_out, N, 1);
        cudaDeviceSynchronize();

        float best = 1e30f;
        for (int trial = 0; trial < 5; trial++) {
            cudaEventRecord(e0);
            read_only<<<blocks, threads>>>(d_data, d_out, N, iters);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }

        double total = (double)bytes * iters;
        double bw = total / (best/1000.0) / 1e9;
        printf("  %-12zu %-12d %-12.2f %-12.0f\n", mb, iters, best, bw);

        cudaFree(d_data); cudaFree(d_out);
    }

    return 0;
}
