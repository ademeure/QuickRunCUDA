// Two blocks coordinate via global flag - measure round-trip latency
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void prod_cons_test(unsigned *flag, unsigned long long *out, int delay_iters) {
    if (blockIdx.x == 0) {
        // Consumer block
        if (threadIdx.x == 0) {
            unsigned long long t0 = clock64();
            while (*((volatile unsigned*)flag) != 42) {}
            unsigned long long t1 = clock64();
            out[0] = t1 - t0;
        }
    } else {
        // Producer block
        if (threadIdx.x == 0) {
            // Spin-delay using clock64
            unsigned long long t0 = clock64();
            while (clock64() - t0 < (unsigned long long)delay_iters * 200) {}
            __threadfence();
            *flag = 42;
            out[1] = clock64();
        }
    }
}

int main() {
    cudaSetDevice(0);
    unsigned *d_flag; cudaMalloc(&d_flag, sizeof(unsigned));
    unsigned long long *d_out; cudaMalloc(&d_out, 16*sizeof(unsigned long long));

    printf("# B300 intra-kernel cross-block flag wait\n");
    printf("# %-15s %-15s %-15s\n", "delay_us", "spin_cyc", "spin_us");

    for (int delay : {0, 100, 500, 1000, 5000, 10000}) {
        cudaMemset(d_flag, 0, sizeof(unsigned));
        prod_cons_test<<<2, 32>>>(d_flag, d_out, delay);
        cudaDeviceSynchronize();
        unsigned long long cyc[2];
        cudaMemcpy(cyc, d_out, 2*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        printf("  %-15d %-15llu %-15.1f\n",
               delay * 200 / 2032,
               (unsigned long long)cyc[0],
               cyc[0] / 2.032 / 1000);
    }

    return 0;
}
