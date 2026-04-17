// Measure cluster.sync() latency on B300
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

extern "C" __global__ void __cluster_dims__(2,1,1) cluster2(unsigned long long *out) {
    auto cluster = cg::this_cluster();
    unsigned long long start, end;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < 10; i++) cluster.sync();
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));

    if (threadIdx.x == 0) out[blockIdx.x] = end - start;
}

extern "C" __global__ void __cluster_dims__(4,1,1) cluster4(unsigned long long *out) {
    auto cluster = cg::this_cluster();
    unsigned long long start, end;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < 10; i++) cluster.sync();
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));

    if (threadIdx.x == 0) out[blockIdx.x] = end - start;
}

extern "C" __global__ void __cluster_dims__(8,1,1) cluster8(unsigned long long *out) {
    auto cluster = cg::this_cluster();
    unsigned long long start, end;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < 10; i++) cluster.sync();
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));

    if (threadIdx.x == 0) out[blockIdx.x] = end - start;
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out;
    cudaMalloc(&d_out, 32 * sizeof(unsigned long long));

    // Warmup + run each
    for (int i = 0; i < 3; i++) {
        cluster2<<<2, 32>>>(d_out);
        cluster4<<<4, 32>>>(d_out);
        cluster8<<<8, 32>>>(d_out);
    }
    cudaDeviceSynchronize();

    auto run_test = [&](const char *name, auto fn, int size) {
        fn<<<size, 32>>>(d_out);
        cudaDeviceSynchronize();
        unsigned long long h[16];
        cudaMemcpy(h, d_out, size * 8, cudaMemcpyDeviceToHost);

        unsigned long long min_c = h[0], max_c = h[0], sum = 0;
        for (int i = 0; i < size; i++) {
            if (h[i] < min_c) min_c = h[i];
            if (h[i] > max_c) max_c = h[i];
            sum += h[i];
        }
        printf("  %-10s : min=%llu max=%llu avg=%llu -> %.1f cy/cluster.sync()\n",
               name, min_c, max_c, sum / size, (sum / (double)size) / 10.0);
    };

    printf("# B300 cluster.sync() latency (10 syncs per kernel)\n\n");
    run_test("cluster=2", cluster2, 2);
    run_test("cluster=4", cluster4, 4);
    run_test("cluster=8", cluster8, 8);

    cudaFree(d_out);
    return 0;
}
