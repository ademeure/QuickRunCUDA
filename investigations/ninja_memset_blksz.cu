// Vary thread/block count for the ninja write recipe
#include <cuda_runtime.h>
#include <cstdio>

template <int TPB>
__launch_bounds__(TPB, 8) __global__ void w_ninja(int *data, int v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *p = data + (warp_id * 32 + lane) * 8;
    asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
        :: "l"(p), "r"(v) : "memory");
}

template <int TPB>
double bench(int *d, size_t bytes, cudaEvent_t e0, cudaEvent_t e1) {
    int blocks = bytes / (TPB * 32);
    for (int i = 0; i < 5; i++) w_ninja<TPB><<<blocks, TPB>>>(d, 0xab);
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(e0);
        w_ninja<TPB><<<blocks, TPB>>>(d, 0xab);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    return bytes / (best/1000) / 1e9;
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 4ull * 1024 * 1024 * 1024;
    int *d; cudaMalloc(&d, bytes);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("# Vary threads/block for ninja write at 4 GB\n");
    printf("# TPB   blocks      GB/s    %% spec\n");

    #define R(TPB) do { double g = bench<TPB>(d, bytes, e0, e1); \
        printf("  %4d  %8d  %6.0f  %.2f%%\n", TPB, (int)(bytes / (TPB * 32)), g, g/7672*100); \
    } while(0)

    R(32);
    R(64);
    R(128);
    R(256);
    R(512);
    R(1024);
    return 0;
}
