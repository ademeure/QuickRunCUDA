// Cleaner overlap test - no cross-iteration dependency leak
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void ffma(float *out, int iters) {
    float a = threadIdx.x * 0.001f, b = a+0.001f, c = b+0.001f, d = c+0.001f;
    for (int i = 0; i < iters; i++) {
        a = a*1.0001f + 0.0001f; b = b*1.0001f + 0.0001f;
        c = c*1.0001f + 0.0001f; d = d*1.0001f + 0.0001f;
    }
    if (a+b+c+d < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 148 * 256 * sizeof(float));
    void *d_buf; cudaMalloc(&d_buf, 4ull * 1024 * 1024 * 1024);

    cudaStream_t s_ffma, s_memset;
    cudaStreamCreateWithFlags(&s_ffma, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_memset, cudaStreamNonBlocking);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    int ffma_iters = 1000000;
    int blocks = 148, threads = 256;

    // Warmup separately
    for (int i = 0; i < 5; i++) {
        ffma<<<blocks, threads, 0, s_ffma>>>(d_out, ffma_iters);
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < 5; i++) {
        cudaMemsetAsync(d_buf, 0xab, 4ull*1024*1024*1024, s_memset);
    }
    cudaDeviceSynchronize();

    // Single isolated FFMA timing
    float ffma_alone_best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0, s_ffma);
        ffma<<<blocks, threads, 0, s_ffma>>>(d_out, ffma_iters);
        cudaEventRecord(e1, s_ffma);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < ffma_alone_best) ffma_alone_best = ms;
    }

    // Single isolated cudaMemset timing
    cudaEvent_t m0, m1; cudaEventCreate(&m0); cudaEventCreate(&m1);
    float memset_alone_best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(m0, s_memset);
        cudaMemsetAsync(d_buf, 0xab, 4ull*1024*1024*1024, s_memset);
        cudaEventRecord(m1, s_memset);
        cudaEventSynchronize(m1);
        float ms; cudaEventElapsedTime(&ms, m0, m1);
        if (ms < memset_alone_best) memset_alone_best = ms;
    }

    // Now overlap test - measure FFMA timing during concurrent cudaMemset
    float ffma_with_memset_best = 1e30f;
    float memset_with_ffma_best = 1e30f;
    for (int trial = 0; trial < 5; trial++) {
        cudaEventRecord(e0, s_ffma);
        ffma<<<blocks, threads, 0, s_ffma>>>(d_out, ffma_iters);
        cudaEventRecord(e1, s_ffma);

        cudaEventRecord(m0, s_memset);
        cudaMemsetAsync(d_buf, 0xab, 4ull*1024*1024*1024, s_memset);
        cudaEventRecord(m1, s_memset);

        cudaEventSynchronize(e1);
        cudaEventSynchronize(m1);
        float ms_f, ms_m;
        cudaEventElapsedTime(&ms_f, e0, e1);
        cudaEventElapsedTime(&ms_m, m0, m1);
        if (ms_f < ffma_with_memset_best) ffma_with_memset_best = ms_f;
        if (ms_m < memset_with_ffma_best) memset_with_ffma_best = ms_m;
    }

    printf("# cudaMemset DMA test - per-stream timing\n\n");
    printf("  FFMA alone:                  %.2f ms\n", ffma_alone_best);
    printf("  cudaMemset alone:            %.2f ms (= %.0f GB/s)\n",
           memset_alone_best, 4.0*1024*1024*1024/(memset_alone_best/1000)/1e9);
    printf("\n  Concurrent (separate streams):\n");
    printf("  FFMA   in concurrent run:    %.2f ms (slowdown %.2fx)\n",
           ffma_with_memset_best, ffma_with_memset_best/ffma_alone_best);
    printf("  Memset in concurrent run:    %.2f ms (slowdown %.2fx)\n",
           memset_with_ffma_best, memset_with_ffma_best/memset_alone_best);

    return 0;
}
