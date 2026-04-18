// NINJA multi-buffer write
#include <cuda_runtime.h>
#include <cstdio>

template <int N_BUFS>
__launch_bounds__(256, 8) __global__ void w_multibuf(int * const *bufs, int v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    #pragma unroll
    for (int b = 0; b < N_BUFS; b++) {
        int *p = bufs[b] + (warp_id * 32 + lane) * 8;
        asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
            :: "l"(p), "r"(v) : "memory");
    }
}

template <int N_BUFS>
double bench(int * const *d_bufs, size_t bytes_per_buf, cudaEvent_t e0, cudaEvent_t e1) {
    int blocks = bytes_per_buf / (256 * 32);
    for (int i = 0; i < 5; i++) w_multibuf<N_BUFS><<<blocks, 256>>>(d_bufs, 0xab);
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(e0);
        w_multibuf<N_BUFS><<<blocks, 256>>>(d_bufs, 0xab);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    return bytes_per_buf * N_BUFS / (best/1000) / 1e9;
}

int main() {
    cudaSetDevice(0);
    size_t bytes_per_buf = 1ull * 1024 * 1024 * 1024;
    int *bufs[8];
    for (int i = 0; i < 8; i++) cudaMalloc(&bufs[i], bytes_per_buf);
    int *d_bufs; cudaMalloc(&d_bufs, 8 * sizeof(int*));
    cudaMemcpy(d_bufs, bufs, 8 * sizeof(int*), cudaMemcpyHostToDevice);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("# Multi-buffer concurrent writes\n");
    printf("# N_BUFS  total_GB  GB/s    %% spec\n");
    #define W(N) do { \
        double g = bench<N>((int* const*)d_bufs, bytes_per_buf, e0, e1); \
        printf("  %2d     %4d     %6.0f  %.2f%%\n", N, N, g, g/7672*100); \
    } while(0)
    W(1); W(2); W(4); W(8);

    return 0;
}
