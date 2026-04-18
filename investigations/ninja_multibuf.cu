// NINJA multi-buffer R: do reads from N separate buffers simultaneously
// Hypothesis: if HBM stacks have independent channels, total R aggregate
// across N buffers > 7.31 TB/s single-buffer ceiling
#include <cuda_runtime.h>
#include <cstdio>

template <int N_BUFS>
__launch_bounds__(256, 8) __global__ void r_multibuf(int * const *bufs, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int acc = 0;
    // Each warp reads 1 v8 from each of N_BUFS buffers
    #pragma unroll
    for (int b = 0; b < N_BUFS; b++) {
        const int *p = bufs[b] + (warp_id * 32 + lane) * 8;
        int r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "l"(p));
        acc ^= r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7;
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
    else if (tid == 0) out[0] = acc;
}

template <int N_BUFS>
double bench(int * const *d_bufs, size_t bytes_per_buf, int *d_out, cudaEvent_t e0, cudaEvent_t e1) {
    int blocks = bytes_per_buf / (256 * 32);  // each warp reads 1 KB from each buffer
    for (int i = 0; i < 5; i++) r_multibuf<N_BUFS><<<blocks, 256>>>(d_bufs, d_out);
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(e0);
        r_multibuf<N_BUFS><<<blocks, 256>>>(d_bufs, d_out);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    size_t total_bytes = bytes_per_buf * N_BUFS;
    return total_bytes / (best/1000) / 1e9;
}

int main() {
    cudaSetDevice(0);
    size_t bytes_per_buf = 1ull * 1024 * 1024 * 1024;  // 1 GB per buffer
    int *bufs[16];
    for (int i = 0; i < 16; i++) {
        cudaMalloc(&bufs[i], bytes_per_buf);
        cudaMemset(bufs[i], 0xab + i, bytes_per_buf);
    }
    int *d_bufs;
    cudaMalloc(&d_bufs, 16 * sizeof(int*));
    cudaMemcpy(d_bufs, bufs, 16 * sizeof(int*), cudaMemcpyHostToDevice);
    int *d_out; cudaMalloc(&d_out, 1<<24);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("# Multi-buffer concurrent reads — does aggregate exceed 7.31 single-buf?\n");
    printf("# N_BUFS  total_GB  GB/s    %% of 7672 spec\n");

    #define R(N) do { \
        double g = bench<N>((int* const*)d_bufs, bytes_per_buf, d_out, e0, e1); \
        printf("  %2d     %4d     %6.0f  %.2f%%\n", N, N, g, g/7672*100); \
    } while(0)

    R(1); R(2); R(4); R(8); R(16);

    return 0;
}
