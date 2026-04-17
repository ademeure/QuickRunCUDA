// L2 cache peak BW (working set fits in L2)
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void __launch_bounds__(512, 4) read_l2(const int4 *data, int *out, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int4 a = make_int4(0,0,0,0), b=a, c=a, d=a, e=a, f=a, g=a, h=a;

    for (int it = 0; it < iters; it++) {
        for (int i = tid; i < N - 7*stride; i += 8 * stride) {
            int4 v0 = data[i];
            int4 v1 = data[i + stride];
            int4 v2 = data[i + 2*stride];
            int4 v3 = data[i + 3*stride];
            int4 v4 = data[i + 4*stride];
            int4 v5 = data[i + 5*stride];
            int4 v6 = data[i + 6*stride];
            int4 v7 = data[i + 7*stride];
            a.x ^= v0.x; b.x ^= v1.x; c.x ^= v2.x; d.x ^= v3.x;
            e.x ^= v4.x; f.x ^= v5.x; g.x ^= v6.x; h.x ^= v7.x;
        }
    }
    int s = a.x ^ b.x ^ c.x ^ d.x ^ e.x ^ f.x ^ g.x ^ h.x;
    if (s == 0xdeadbeef) out[tid] = s;
}

int main() {
    cudaSetDevice(0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148, threads = 512;

    printf("# B300 L2 vs HBM bandwidth (8-ILP read pattern)\n");
    printf("# %-12s %-12s %-12s %-12s\n", "size_MB", "fits", "iters", "BW_GB/s");

    for (size_t mb : {16, 64, 100, 126, 256, 1024}) {
        size_t bytes = mb * 1024 * 1024;
        int N = bytes / 16;
        int iters = mb < 128 ? 50 : (mb == 256 ? 20 : 5);

        int4 *d_data; cudaMalloc(&d_data, bytes);
        cudaMemset(d_data, 0xab, bytes);
        int *d_out; cudaMalloc(&d_out, blocks * threads * sizeof(int));

        for (int i = 0; i < 3; i++) read_l2<<<blocks, threads>>>(d_data, d_out, N, 1);
        cudaDeviceSynchronize();

        float best = 1e30f;
        for (int trial = 0; trial < 5; trial++) {
            cudaEventRecord(e0);
            read_l2<<<blocks, threads>>>(d_data, d_out, N, iters);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }

        double total = (double)bytes * iters;
        double bw = total / (best/1000.0) / 1e9;
        const char *fits = (mb <= 64) ? "L2(easily)" : (mb <= 126) ? "L2(edge)" : "DRAM";
        printf("  %-12zu %-12s %-12d %-12.0f\n", mb, fits, iters, bw);

        cudaFree(d_data); cudaFree(d_out);
    }

    return 0;
}
