// Two blocks ping-pong - simplified protocol
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void pingpong(unsigned *flag_a, unsigned *flag_b, unsigned long long *out, int rounds) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Initiator: write flag_a, wait for flag_b
        unsigned long long t0 = clock64();
        for (int i = 0; i < rounds; i++) {
            __threadfence();
            *flag_a = i + 1;
            // Wait for response
            while (*((volatile unsigned*)flag_b) != i + 1) {}
        }
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    } else if (blockIdx.x == 1 && threadIdx.x == 0) {
        // Responder: wait for flag_a, write flag_b
        for (int i = 0; i < rounds; i++) {
            while (*((volatile unsigned*)flag_a) != i + 1) {}
            __threadfence();
            *flag_b = i + 1;
        }
    }
}

int main() {
    cudaSetDevice(0);
    unsigned *d_a; cudaMalloc(&d_a, sizeof(unsigned));
    unsigned *d_b; cudaMalloc(&d_b, sizeof(unsigned));
    unsigned long long *d_out; cudaMalloc(&d_out, 16*sizeof(unsigned long long));

    int rounds = 1000;
    cudaMemset(d_a, 0, sizeof(unsigned));
    cudaMemset(d_b, 0, sizeof(unsigned));
    pingpong<<<2, 32>>>(d_a, d_b, d_out, rounds);
    cudaDeviceSynchronize();

    unsigned long long cyc;
    cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
    double per = (double)cyc / rounds;
    printf("# B300 cross-block ping-pong (%d rounds, separate flags)\n", rounds);
    printf("  Total: %llu cyc = %.0f us\n", (unsigned long long)cyc, cyc/2.032/1000);
    printf("  Per round-trip: %.1f cyc = %.0f ns (= %.0f ns one-way)\n",
           per, per/2.032, per/2.032/2);

    return 0;
}
