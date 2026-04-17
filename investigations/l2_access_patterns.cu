// L2 cache behavior under different access patterns
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void seq_read(const float *data, float *out, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float a = 0;
    for (int it = 0; it < iters; it++) {
        for (int i = tid; i < N; i += stride) a += data[i];
    }
    if (a < -1e30f) out[tid] = a;
}

extern "C" __global__ void random_read(const float *data, float *out, int N, int iters, unsigned seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0;
    unsigned s = tid * 2654435761u + seed;
    for (int it = 0; it < iters * 64; it++) {
        s = s * 1664525u + 1013904223u;
        int idx = s & (N - 1);
        a += data[idx];
    }
    if (a < -1e30f) out[tid] = a;
}

extern "C" __global__ void strided_read(const float *data, float *out, int N, int iters, int stride_floats) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int max_strides = N / stride_floats;
    float a = 0;
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < max_strides; i++) {
            int idx = (tid + i * stride_floats) & (N - 1);
            a += data[idx];
        }
    }
    if (a < -1e30f) out[tid] = a;
}

int main() {
    cudaSetDevice(0);

    // Sizes spanning L1/L2/HBM
    int sizes_mb[] = {1, 16, 64, 126, 256, 1024};
    int iters_per[] = {500, 100, 50, 20, 10, 3};
    const char *fits[] = {"L1", "L2", "L2", "L2(edge)", "DRAM", "DRAM"};

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148, threads = 256;
    float *d_out; cudaMalloc(&d_out, blocks * threads * sizeof(float));

    auto bench = [&](auto launch) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 3; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("# B300 read patterns vs working-set size\n");
    printf("# %-12s %-10s %-12s %-12s %-12s\n",
           "size_MB", "fits", "seq_GB/s", "random_GB/s", "strided4K");

    for (int s = 0; s < 6; s++) {
        size_t bytes = (size_t)sizes_mb[s] * 1024 * 1024;
        int N = bytes / 4;
        int iters = iters_per[s];

        float *d_data; cudaMalloc(&d_data, bytes);
        cudaMemset(d_data, 0, bytes);

        float t_seq = bench([&]{ seq_read<<<blocks, threads>>>(d_data, d_out, N, iters); });
        float t_rnd = bench([&]{ random_read<<<blocks, threads>>>(d_data, d_out, N, iters, 1u); });
        float t_str = bench([&]{ strided_read<<<blocks, threads>>>(d_data, d_out, N, iters, 1024); });

        // Effective bytes: sequential reads N × iters × 4B
        double bw_seq = (double)N * iters * 4 / (t_seq/1000) / 1e9;
        double bw_rnd = (double)blocks * threads * iters * 64 * 4 / (t_rnd/1000) / 1e9;
        double bw_str = (double)blocks * threads * (N/1024) * iters * 4 / (t_str/1000) / 1e9;

        printf("  %-12d %-10s %-12.0f %-12.0f %-12.0f\n",
               sizes_mb[s], fits[s], bw_seq, bw_rnd, bw_str);

        cudaFree(d_data);
    }

    return 0;
}
