// Cross-GPU atomic ops latency
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void atom_loop(unsigned *target, unsigned long long *out, int iters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        for (int i = 0; i < iters; i++) {
            atomicAdd(target, 1);
        }
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    }
}

int main() {
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);

    cudaSetDevice(1);
    unsigned *d1; cudaMalloc(&d1, sizeof(unsigned));
    cudaMemset(d1, 0, sizeof(unsigned));

    cudaSetDevice(0);
    unsigned *d0; cudaMalloc(&d0, sizeof(unsigned));
    cudaMemset(d0, 0, sizeof(unsigned));
    unsigned long long *d_out; cudaMalloc(&d_out, sizeof(unsigned long long));

    int iters = 1000;

    // Warmup
    atom_loop<<<1, 32>>>(d0, d_out, 100);
    cudaDeviceSynchronize();

    // Local atomic on GPU 0
    atom_loop<<<1, 32>>>(d0, d_out, iters);
    cudaDeviceSynchronize();
    unsigned long long cyc;
    cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
    printf("# Local atomicAdd from GPU0 to GPU0 mem: %.1f cyc = %.2f ns\n",
           (double)cyc/iters, (double)cyc/iters/2.032);

    // Remote atomic
    atom_loop<<<1, 32>>>(d1, d_out, iters);
    cudaError_t err = cudaDeviceSynchronize();
    if (err) {
        printf("# Remote atomic FAILED: %s\n", cudaGetErrorString(err));
    } else {
        cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        printf("# Remote atomicAdd from GPU0 to GPU1 mem: %.1f cyc = %.2f ns\n",
               (double)cyc/iters, (double)cyc/iters/2.032);
    }

    return 0;
}
