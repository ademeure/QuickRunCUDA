// V10: Streaming BW at various working set sizes (independent loads, no chain)
#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);} }while(0)

#ifndef K_INNER
#define K_INNER 128
#endif

__global__ void stream_read(float4* A, float4* out, int buf_fl4_mask, int iters) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    float4 acc0 = make_float4(0,0,0,0);
    float4 acc1 = make_float4(0,0,0,0);

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        #pragma unroll 8
        for (int k = 0; k < K_INNER; k++) {
            int idx = (gtid + k * 37888 + i * 128) & buf_fl4_mask;
            float4 v = A[idx];
            if (k & 1) { acc1.x += v.x; acc1.y += v.y; acc1.z += v.z; acc1.w += v.w; }
            else       { acc0.x += v.x; acc0.y += v.y; acc0.z += v.z; acc0.w += v.w; }
        }
    }
    if (acc0.x + acc1.x == 1.234567e-30f) out[gtid] = acc0;
}

int main() {
    cudaSetDevice(0);

    // Buffer sizes to test (in MB, as float4 count)
    struct { size_t mb; const char* label; } sizes[] = {
        {1, "1 MB (L1 hit)"},
        {16, "16 MB (> L1, L2 fit)"},
        {64, "64 MB (half L2)"},
        {128, "128 MB (~L2)"},
        {256, "256 MB (> L2, DRAM)"},
    };

    printf("=== Streaming BW (independent indexed loads) ===\n");
    for (auto& sz : sizes) {
        size_t n_fl4 = sz.mb * 1024 * 1024 / 16;
        // Round to power of 2
        size_t pow2 = 1;
        while (pow2 * 2 <= n_fl4) pow2 *= 2;
        int mask = (int)(pow2 - 1);

        float4* d_A;
        float4* d_out;
        CK(cudaMalloc(&d_A, pow2 * 16));
        CK(cudaMalloc(&d_out, 148 * 256 * 16));

        int iters = 500;
        int blocks = 148;
        int threads = 256;

        // Warmup
        stream_read<<<blocks, threads>>>(d_A, d_out, mask, iters);
        cudaDeviceSynchronize();

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        stream_read<<<blocks, threads>>>(d_A, d_out, mask, iters);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);

        // Bytes read: blocks × threads × iters × K_INNER × 16 B
        double bytes = (double)blocks * threads * iters * K_INNER * 16;
        double tbs = bytes / 1e12 / (ms / 1000.0);

        printf("  %-24s (pow2=%zu fl4): %.2f ms → %.2f TB/s\n",
            sz.label, pow2, ms, tbs);

        cudaFree(d_A); cudaFree(d_out);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    return 0;
}
