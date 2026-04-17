// Test: can we skip MEMBAR.ALL.GPU and just use barrier_arrive+wait?
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2,1,1) cg_sync(unsigned long long *out, int iters) {
    auto cluster = cg::this_cluster();
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) cluster.sync();
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = t1 - t0;
}

__global__ void __cluster_dims__(2,1,1) raw_barrier(unsigned long long *out, int iters) {
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        // Lower-level: just barrier arrive + wait, no MEMBAR
        asm volatile("barrier.cluster.arrive.aligned;");
        asm volatile("barrier.cluster.wait.aligned;");
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = t1 - t0;
}

__global__ void __cluster_dims__(2,1,1) raw_barrier_relaxed(unsigned long long *out, int iters) {
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        // Even more relaxed - relaxed.aligned variant
        asm volatile("barrier.cluster.arrive.relaxed.aligned;");
        asm volatile("barrier.cluster.wait.aligned;");
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = t1 - t0;
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out; cudaMalloc(&d_out, 16 * sizeof(unsigned long long));

    int iters = 1000;

    auto run = [&](auto launch, const char *name) {
        launch();
        cudaError_t err = cudaDeviceSynchronize();
        if (err) { printf("  %-40s ERR: %s\n", name, cudaGetErrorString(err)); return; }
        unsigned long long cyc; cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  %-40s %.1f cyc = %.1f ns\n", name, per, per/2.032);
    };

    printf("# B300 cluster sync variants (cluster size 2)\n\n");

    run([&]{ cg_sync<<<2, 32>>>(d_out, iters); }, "cg::cluster.sync()");
    run([&]{ raw_barrier<<<2, 32>>>(d_out, iters); }, "barrier.cluster.arrive+wait");
    run([&]{ raw_barrier_relaxed<<<2, 32>>>(d_out, iters); }, "barrier.cluster.arrive.relaxed+wait");

    return 0;
}
