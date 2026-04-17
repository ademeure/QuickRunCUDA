// cooperative_groups reduce vs raw shfl reduce
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cstdio>

namespace cg = cooperative_groups;

__global__ void cg_reduce(unsigned *out, int iters) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    unsigned a = threadIdx.x + 1;
    for (int i = 0; i < iters; i++) {
        a = cg::reduce(warp, a, cg::plus<unsigned>());
        a += i;
    }
    if (a == 0xdeadbeef) out[blockIdx.x] = a;
}

__global__ void shfl_reduce(unsigned *out, int iters) {
    unsigned a = threadIdx.x + 1;
    for (int i = 0; i < iters; i++) {
        a += __shfl_xor_sync(0xffffffff, a, 1);
        a += __shfl_xor_sync(0xffffffff, a, 2);
        a += __shfl_xor_sync(0xffffffff, a, 4);
        a += __shfl_xor_sync(0xffffffff, a, 8);
        a += __shfl_xor_sync(0xffffffff, a, 16);
        a += i;
    }
    if (a == 0xdeadbeef) out[blockIdx.x] = a;
}

__global__ void cg_inclusive_scan(unsigned *out, int iters) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    unsigned a = threadIdx.x + 1;
    for (int i = 0; i < iters; i++) {
        a = cg::inclusive_scan(warp, a, cg::plus<unsigned>());
        a += i;
    }
    if (a == 0xdeadbeef) out[blockIdx.x] = a;
}

int main() {
    cudaSetDevice(0);
    unsigned *d_out; cudaMalloc(&d_out, 1024 * sizeof(unsigned));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 128;

    auto bench = [&](auto launch, const char *name) {
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
        printf("  %-30s %.3f ms\n", name, best);
    };

    printf("# B300 cooperative_groups vs raw warp-level reductions\n");
    printf("# 148 × 128 thr × 100k iter\n\n");

    bench([&]{ shfl_reduce<<<blocks, threads>>>(d_out, iters); }, "raw shfl_xor reduce");
    bench([&]{ cg_reduce<<<blocks, threads>>>(d_out, iters); }, "cg::reduce(warp, +)");
    bench([&]{ cg_inclusive_scan<<<blocks, threads>>>(d_out, iters); }, "cg::inclusive_scan");

    return 0;
}
