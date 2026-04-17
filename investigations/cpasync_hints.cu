#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

__global__ void cpasync_ca(const int4 *src, int *out, int N, int iters) {
    extern __shared__ int4 smem[];
    int tid = threadIdx.x;
    if (tid >= 32) return;

    uint32_t s_addr = __cvta_generic_to_shared(&smem[tid]);
    int sum = 0;
    for (int i = 0; i < iters; i++) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(s_addr), "l"(src + ((tid + i) & (N-1))));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        sum += smem[tid].x;
    }
    if (sum < -1) out[blockIdx.x] = sum;
}

__global__ void cpasync_cg(const int4 *src, int *out, int N, int iters) {
    extern __shared__ int4 smem[];
    int tid = threadIdx.x;
    if (tid >= 32) return;

    uint32_t s_addr = __cvta_generic_to_shared(&smem[tid]);
    int sum = 0;
    for (int i = 0; i < iters; i++) {
        // .cg = cache global only (skip L1)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(s_addr), "l"(src + ((tid + i) & (N-1))));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        sum += smem[tid].x;
    }
    if (sum < -1) out[blockIdx.x] = sum;
}

int main() {
    cudaSetDevice(0);
    int N = 1024;  // small - L2-resident
    int4 *d_src; cudaMalloc(&d_src, N * sizeof(int4));
    cudaMemset(d_src, 1, N * sizeof(int4));
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int iters = 100000;
    int blocks = 148, threads = 32;

    auto bench = [&](auto launch, const char *name) {
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
        long total = (long)blocks * 32 * iters * 16;
        double tb = total / (best/1000.0) / 1e12;
        printf("  %-25s %.3f ms  %.2f TB/s\n", name, best, tb);
    };

    printf("# B300 cp.async cache hint variants (small N L2-resident)\n\n");
    bench([&]{ cpasync_ca<<<blocks, threads, 32*sizeof(int4)>>>(d_src, d_out, N, iters); }, "cp.async.ca (cache all)");
    bench([&]{ cpasync_cg<<<blocks, threads, 32*sizeof(int4)>>>(d_src, d_out, N, iters); }, "cp.async.cg (skip L1)");

    return 0;
}
