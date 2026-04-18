// __syncthreads() cost vs block size on B300
//
// Theoretical:
//   __syncthreads compiles to BAR.SYNC.DEFER 0
//   Cost should be ~independent of block size up to a point (warp count drives it)
//   Catalog says 5-10 ns but no sweep
//
// Method: tight loop of N __syncthreads() inside kernel; measure per-call cost
#include <cuda_runtime.h>
#include <cstdio>

template <int BS>
__launch_bounds__(BS, 8) __global__ void sync_loop(long long *out, int N) {
    long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        __syncthreads();
    }
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = t1 - t0;
}

template <int BS>
double bench(int N) {
    long long *d_out; cudaMalloc(&d_out, 16);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 4;  // moderate occupancy
    for (int i = 0; i < 3; i++) sync_loop<BS><<<blocks, BS>>>(d_out, N);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("bs=%d: error %s\n", BS, cudaGetErrorString(cudaGetLastError()));
        return 0;
    }
    long long h;
    sync_loop<BS><<<blocks, BS>>>(d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h, d_out, 8, cudaMemcpyDeviceToHost);
    double cy_per_sync = (double)h / N;
    double ns_per_sync = cy_per_sync / 2.032;
    printf("  bs=%-5d (warps=%2d): %lld total cy / %d syncs = %.2f cy/sync = %.2f ns\n",
           BS, BS/32, h, N, cy_per_sync, ns_per_sync);
    cudaFree(d_out);
    return cy_per_sync;
}

int main() {
    cudaSetDevice(0);
    int N = 10000;
    printf("# __syncthreads cost vs block size (clock64 in-kernel)\n");
    bench<32>(N);
    bench<64>(N);
    bench<128>(N);
    bench<256>(N);
    bench<512>(N);
    bench<1024>(N);
    return 0;
}
