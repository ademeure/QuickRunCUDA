#include <cuda_runtime.h>
#include <cstdio>
template <int BS>
__launch_bounds__(BS, 8) __global__ void sync_loop(int *out, int N) {
    int x = threadIdx.x;
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        __syncthreads();
        x ^= i;
    }
    if (x == 0xdeadbeef) out[0] = x;
}
template <int BS>
double bench_wall(int N) {
    int *d_out; cudaMalloc(&d_out, 4);
    int blocks = 148 * 4;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) sync_loop<BS><<<blocks, BS>>>(d_out, N);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) return 0;
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        sync_loop<BS><<<blocks, BS>>>(d_out, N);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    double ns_per_sync = (double)best * 1e6 / N;
    printf("  bs=%d wall-clock: %.4f ms / %d = %.2f ns/sync\n", BS, best, N, ns_per_sync);
    return ns_per_sync;
}
int main() {
    cudaSetDevice(0);
    int N = 100000;
    bench_wall<32>(N);
    bench_wall<64>(N);
    bench_wall<128>(N);
    bench_wall<256>(N);
    bench_wall<512>(N);
    bench_wall<1024>(N);
    return 0;
}
