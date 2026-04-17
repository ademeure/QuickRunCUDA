#include <cuda_runtime.h>
#include <cstdio>

// Force real serialization by using the result as part of the value
__global__ void atom_serial(unsigned *target, unsigned long long *out, int iters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        unsigned val = 1;
        for (int i = 0; i < iters; i++) {
            val = atomicAdd(target, val) + 1;  // val depends on old
        }
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
        if (val == 0xdead) target[1] = val;
    }
}

int main() {
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);

    cudaSetDevice(1);
    unsigned *d1; cudaMalloc(&d1, 8 * sizeof(unsigned));
    cudaMemset(d1, 0, 8 * sizeof(unsigned));

    cudaSetDevice(0);
    unsigned *d0; cudaMalloc(&d0, 8 * sizeof(unsigned));
    cudaMemset(d0, 0, 8 * sizeof(unsigned));
    unsigned long long *d_out; cudaMalloc(&d_out, sizeof(unsigned long long));

    int iters = 1000;

    atom_serial<<<1, 32>>>(d0, d_out, iters);
    cudaDeviceSynchronize();
    unsigned long long cyc;
    cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
    printf("# Local true-serial atomic: %.1f cyc = %.2f ns\n",
           (double)cyc/iters, (double)cyc/iters/2.032);

    atom_serial<<<1, 32>>>(d1, d_out, iters);
    cudaError_t err = cudaDeviceSynchronize();
    if (err) {
        printf("# Remote FAILED: %s\n", cudaGetErrorString(err));
    } else {
        cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        printf("# Remote true-serial atomic: %.1f cyc = %.2f ns\n",
               (double)cyc/iters, (double)cyc/iters/2.032);
    }

    return 0;
}
