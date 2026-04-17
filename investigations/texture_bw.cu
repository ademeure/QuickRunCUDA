// B300 texture cache vs L1 cache - is texture path still a thing?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void via_texture(cudaTextureObject_t tex, float *out, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0;
    for (int i = 0; i < iters; i++) {
        for (int j = 0; j < N; j += 32) {
            int idx = (tid + j) & (N - 1);
            a += tex1Dfetch<float>(tex, idx);
        }
    }
    if (a < -1e30f) out[tid] = a;
}

extern "C" __global__ void via_global(const float * __restrict__ data, float *out, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0;
    for (int i = 0; i < iters; i++) {
        for (int j = 0; j < N; j += 32) {
            int idx = (tid + j) & (N - 1);
            a += __ldg(data + idx);
        }
    }
    if (a < -1e30f) out[tid] = a;
}

extern "C" __global__ void via_global_normal(const float *data, float *out, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0;
    for (int i = 0; i < iters; i++) {
        for (int j = 0; j < N; j += 32) {
            int idx = (tid + j) & (N - 1);
            a += data[idx];
        }
    }
    if (a < -1e30f) out[tid] = a;
}

int main() {
    cudaSetDevice(0);

    // Test sizes that fit in L1, L2, and miss to DRAM
    printf("# B300 texture vs __ldg vs plain global - read latency/throughput\n");
    printf("# Pattern: cycle through array in 32-element steps with ILP\n\n");
    printf("# %-12s %-12s %-12s %-12s %-12s %-12s\n",
           "size", "fits", "iters", "tex_GB/s", "ldg_GB/s", "global_GB/s");

    int blocks = 148, threads = 256;
    float *d_out; cudaMalloc(&d_out, blocks * threads * sizeof(float));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](auto launch) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int i = 0; i < 5; i++) launch();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        return ms / 5;
    };

    int sizes_kb[] = {16, 64, 256, 1024, 4096, 16384};  // 16K to 16M floats = 64KB to 64MB
    int iters_per[] = {2000, 1000, 500, 200, 50, 10};
    const char *fits[] = {"L1", "L1", "L2", "L2", "L2(?)", "DRAM"};

    for (int s = 0; s < 6; s++) {
        int N = sizes_kb[s] * 1024 / 4;  // floats
        int iters = iters_per[s];
        float *d_data;
        cudaMalloc(&d_data, N * sizeof(float));
        cudaMemset(d_data, 0, N * sizeof(float));

        // Build texture
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = d_data;
        resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        resDesc.res.linear.sizeInBytes = N * sizeof(float);
        cudaTextureDesc texDesc = {};
        texDesc.readMode = cudaReadModeElementType;
        cudaTextureObject_t tex;
        cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);

        float t_tex = bench([&]{ via_texture<<<blocks, threads>>>(tex, d_out, N, iters); });
        float t_ldg = bench([&]{ via_global<<<blocks, threads>>>(d_data, d_out, N, iters); });
        float t_glo = bench([&]{ via_global_normal<<<blocks, threads>>>(d_data, d_out, N, iters); });

        // Bytes per kernel = blocks * threads * iters * (N/32) * 4
        double bytes = (double)blocks * threads * iters * (N/32) * 4.0;
        double bw_tex = bytes / (t_tex/1000) / 1e9;
        double bw_ldg = bytes / (t_ldg/1000) / 1e9;
        double bw_glo = bytes / (t_glo/1000) / 1e9;

        printf("  %-12d %-12s %-12d %-12.0f %-12.0f %-12.0f\n",
               sizes_kb[s], fits[s], iters, bw_tex, bw_ldg, bw_glo);

        cudaDestroyTextureObject(tex);
        cudaFree(d_data);
    }

    cudaFree(d_out);
    return 0;
}
