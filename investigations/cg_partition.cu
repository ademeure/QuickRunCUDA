// cg::tiled_partition cost
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>

namespace cg = cooperative_groups;

__global__ void tile32_test(unsigned *out, int iters) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    unsigned a = threadIdx.x + 1;
    for (int i = 0; i < iters; i++) {
        a = cg::reduce(warp, a, cg::plus<unsigned>());
        a += i;
    }
    if (a == 0xdeadbeef) out[blockIdx.x] = a;
}

__global__ void tile16_test(unsigned *out, int iters) {
    auto block = cg::this_thread_block();
    auto half_warp = cg::tiled_partition<16>(block);

    unsigned a = threadIdx.x + 1;
    for (int i = 0; i < iters; i++) {
        a = cg::reduce(half_warp, a, cg::plus<unsigned>());
        a += i;
    }
    if (a == 0xdeadbeef) out[blockIdx.x] = a;
}

__global__ void tile8_test(unsigned *out, int iters) {
    auto block = cg::this_thread_block();
    auto octet = cg::tiled_partition<8>(block);

    unsigned a = threadIdx.x + 1;
    for (int i = 0; i < iters; i++) {
        a = cg::reduce(octet, a, cg::plus<unsigned>());
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

    printf("# B300 cg::tiled_partition reduce performance\n\n");
    bench([&]{ tile32_test<<<blocks, threads>>>(d_out, iters); }, "tiled_partition<32> + reduce");
    bench([&]{ tile16_test<<<blocks, threads>>>(d_out, iters); }, "tiled_partition<16> + reduce");
    bench([&]{ tile8_test<<<blocks, threads>>>(d_out, iters); }, "tiled_partition<8> + reduce");

    return 0;
}
