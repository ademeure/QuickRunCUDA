// Direct kernel access to peer GPU memory via NVLink
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__launch_bounds__(512, 4) __global__ void read_peer(const int4 *src, int *out, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int4 a = make_int4(0,0,0,0), b=a, c=a, d=a, e=a, f=a, g=a, h=a;
    for (int it = 0; it < iters; it++) {
        for (int i = tid; i < N - 7*stride; i += 8 * stride) {
            int4 v0 = src[i];
            int4 v1 = src[i + stride];
            int4 v2 = src[i + 2*stride];
            int4 v3 = src[i + 3*stride];
            int4 v4 = src[i + 4*stride];
            int4 v5 = src[i + 5*stride];
            int4 v6 = src[i + 6*stride];
            int4 v7 = src[i + 7*stride];
            a.x ^= v0.x; b.x ^= v1.x; c.x ^= v2.x; d.x ^= v3.x;
            e.x ^= v4.x; f.x ^= v5.x; g.x ^= v6.x; h.x ^= v7.x;
        }
    }
    int s = a.x ^ b.x ^ c.x ^ d.x ^ e.x ^ f.x ^ g.x ^ h.x;
    if (s == 0xdeadbeef) out[tid] = s;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);

    // Allocate buffers: src on GPU 1, dst (output) on GPU 0
    cudaSetDevice(1);
    int4 *d1; cudaMalloc(&d1, 1024ull * 1024 * 1024);  // 1 GB on GPU 1
    cudaMemset(d1, 0xab, 1024ull * 1024 * 1024);

    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 148 * 512 * sizeof(int));
    int4 *d0_local; cudaMalloc(&d0_local, 1024ull * 1024 * 1024);
    cudaMemset(d0_local, 0xab, 1024ull * 1024 * 1024);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int N = (1024ull * 1024 * 1024) / 16;  // int4 = 16 B
    int iters = 1;
    int blocks = 148, threads = 512;

    auto bench = [&](auto launch) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("# B300 direct kernel P2P access (GPU0 reads GPU1's memory via NVLink)\n\n");

    // Local read on GPU 0 from its own HBM
    {
        float t = bench([&]{ read_peer<<<blocks, threads>>>(d0_local, d_out, N, iters); });
        printf("  Local HBM read (1 GB):     %.2f ms = %.0f GB/s\n",
               t, (double)1024*1024*1024/(t/1000)/1e9);
    }

    // P2P read GPU0 from GPU1
    {
        float t = bench([&]{ read_peer<<<blocks, threads>>>(d1, d_out, N, iters); });
        printf("  Peer HBM read (1 GB):      %.2f ms = %.0f GB/s\n",
               t, (double)1024*1024*1024/(t/1000)/1e9);
    }

    return 0;
}
