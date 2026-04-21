// V5 K3: Dynamic parallelism cost — kernel launches kernel
#include <cuda_runtime.h>
#include <cstdio>

__global__ void child(unsigned long long* time_out, int idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
        time_out[idx] = t;
    }
}

__global__ void parent_dp(unsigned long long* parent_clock,
                          unsigned long long* child_clock,
                          int n_launches) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    parent_clock[0] = t0;

    for (int i = 0; i < n_launches; i++) {
        child<<<1, 1>>>(child_clock, i);
    }
    // Note: CDP2 removes cudaDeviceSynchronize from device code.
    // Just measure the LAUNCH ISSUE cost (not completion).

    unsigned long long t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    parent_clock[1] = t1;
}

int main() {
    cudaSetDevice(0);
    unsigned long long *parent_clock, *child_clock;
    int N = 100;
    cudaMallocManaged(&parent_clock, 2 * sizeof(unsigned long long));
    cudaMallocManaged(&child_clock, N * sizeof(unsigned long long));

    parent_dp<<<1, 1>>>(parent_clock, child_clock, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    unsigned long long total_cy = parent_clock[1] - parent_clock[0];
    double cy_per_launch = (double)total_cy / N;
    double us_per_launch = cy_per_launch / 1500.0;  // 1500 MHz lock = 1.5 cy/ns

    printf("Dynamic parallelism (kernel launches kernel):\n");
    printf("  N launches: %d\n", N);
    printf("  Total cy: %llu\n", total_cy);
    printf("  Cy per child launch: %.0f\n", cy_per_launch);
    printf("  Time per child launch: %.2f us\n", us_per_launch);

    return 0;
}
