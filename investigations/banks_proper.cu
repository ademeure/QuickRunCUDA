// SHMEM bank conflict cost — proper anti-DCE
#include <cuda_runtime.h>
#include <cstdio>

template<int STRIDE>
__global__ void smem_strided(int *out, int iters, int seed) {
    __shared__ int smem[2048];
    int tid = threadIdx.x;
    smem[tid] = tid + seed;
    smem[tid + 1024] = tid + 1024 + seed;
    __syncthreads();

    int a = 0, b = 0, c = 0, d = 0;
    int base_idx = tid * STRIDE;

    for (int i = 0; i < iters; i++) {
        int idx = (base_idx + i) & 2047;
        a += smem[idx];
        b += smem[(idx + 4) & 2047];
        c += smem[(idx + 8) & 2047];
        d += smem[(idx + 12) & 2047];
        smem[(idx + tid) & 2047] = a + b + c + d;  // write back to defeat DCE
    }
    if (a+b+c+d == 0xdeadbeef) out[blockIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 10000;
    int blocks = 148, threads = 128;

    auto bench = [&](auto launch, int stride) {
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
        long ops = (long)blocks * threads * iters * 4;
        double gops = ops / (best/1000.0) / 1e9;
        printf("  stride=%-3d (%-25s): %.3f ms = %.0f Gops/s\n",
               stride,
               stride == 1 ? "no conflict" :
               stride == 32 ? "32-way conflict" : "various",
               best, gops);
    };

    printf("# B300 SHMEM bank conflict (4-read + 1-write per iter, anti-DCE)\n\n");

    bench([&]{ smem_strided<1><<<blocks, threads>>>(d_out, iters, 1); }, 1);
    bench([&]{ smem_strided<2><<<blocks, threads>>>(d_out, iters, 1); }, 2);
    bench([&]{ smem_strided<4><<<blocks, threads>>>(d_out, iters, 1); }, 4);
    bench([&]{ smem_strided<8><<<blocks, threads>>>(d_out, iters, 1); }, 8);
    bench([&]{ smem_strided<16><<<blocks, threads>>>(d_out, iters, 1); }, 16);
    bench([&]{ smem_strided<32><<<blocks, threads>>>(d_out, iters, 1); }, 32);
    bench([&]{ smem_strided<33><<<blocks, threads>>>(d_out, iters, 1); }, 33);

    return 0;
}
