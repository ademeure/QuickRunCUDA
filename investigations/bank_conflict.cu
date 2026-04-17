// Shared memory bank conflict cost on B300
// 32 banks × 4-byte × 2.032 GHz × 148 SMs = 38.5 TB/s peak
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

template<int STRIDE>
__global__ void smem_read(unsigned *out, int iters) {
    __shared__ unsigned smem[2048];
    int tid = threadIdx.x;
    smem[tid] = tid;
    if (tid + 1024 < 2048) smem[tid + 1024] = tid + 1024;
    __syncthreads();

    unsigned r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    int idx = tid * STRIDE;
    for (int i = 0; i < iters; i++) {
        r0 += smem[(idx + 0) & 2047];
        r1 += smem[(idx + 4) & 2047];
        r2 += smem[(idx + 8) & 2047];
        r3 += smem[(idx + 12) & 2047];
        idx ^= 1;  // anti-DCE
    }
    if (r0+r1+r2+r3 == 0xdeadbeef) out[blockIdx.x] = r0+r1+r2+r3;
}

int main() {
    cudaSetDevice(0);
    unsigned *d_out; cudaMalloc(&d_out, 1024*sizeof(unsigned));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 128;
    long total_ops = (long)blocks * threads * iters * 4;

    auto run = [&](auto launch, int stride, const char *name) {
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
        double bytes = total_ops * 4.0;
        double tb = bytes / (best/1000.0) / 1e12;
        printf("  stride=%-3d (%-25s): %7.3f ms  %5.1f TB/s aggregate\n",
               stride, name, best, tb);
    };

    printf("# B300 shared memory access patterns vs bank conflict\n");
    printf("# 32 banks × 4 B × 148 SMs × 2.032 GHz = 38.5 TB/s theoretical peak\n");
    printf("# 4 reads/iter × 100k iter × 18944 threads = 7.6e9 reads\n\n");

    run([&]{ smem_read<1><<<blocks, threads>>>(d_out, iters); },   1, "no conflict");
    run([&]{ smem_read<2><<<blocks, threads>>>(d_out, iters); },   2, "2-way conflict");
    run([&]{ smem_read<4><<<blocks, threads>>>(d_out, iters); },   4, "4-way conflict");
    run([&]{ smem_read<8><<<blocks, threads>>>(d_out, iters); },   8, "8-way conflict");
    run([&]{ smem_read<16><<<blocks, threads>>>(d_out, iters); }, 16, "16-way conflict");
    run([&]{ smem_read<32><<<blocks, threads>>>(d_out, iters); }, 32, "32-way conflict (worst)");
    run([&]{ smem_read<33><<<blocks, threads>>>(d_out, iters); }, 33, "33 (back to no conflict)");

    return 0;
}
