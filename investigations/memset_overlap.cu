// Test cudaMemset overlap with FULL-OCCUPANCY FFMA and TINY FFMA
#include <cuda_runtime.h>
#include <cstdio>

// Full-occupancy FFMA: 1024 threads/block × 2 blocks/SM = 2048 thr/SM (max)
__launch_bounds__(1024, 2) __global__ void ffma_max_occ(float *out, int iters) {
    float a = threadIdx.x * 0.001f, b = a+0.001f, c = b+0.001f, d = c+0.001f;
    for (int i = 0; i < iters; i++) {
        a = a*1.0001f + 0.0001f; b = b*1.0001f + 0.0001f;
        c = c*1.0001f + 0.0001f; d = d*1.0001f + 0.0001f;
    }
    if (a+b+c+d < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = a+b+c+d;
}

// Tiny FFMA: 128 threads × 1 block, fits in ~1/8 of one SM
__global__ void ffma_tiny(float *out, int iters) {
    float a = threadIdx.x * 0.001f, b = a+0.001f, c = b+0.001f, d = c+0.001f;
    for (int i = 0; i < iters; i++) {
        a = a*1.0001f + 0.0001f; b = b*1.0001f + 0.0001f;
        c = c*1.0001f + 0.0001f; d = d*1.0001f + 0.0001f;
    }
    if (a+b+c+d < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 1024 * 1024 * sizeof(float));
    void *d_buf; cudaMalloc(&d_buf, 4ull * 1024 * 1024 * 1024);

    cudaStream_t s_f, s_m;
    cudaStreamCreateWithFlags(&s_f, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_m, cudaStreamNonBlocking);

    cudaEvent_t fe0, fe1, me0, me1;
    cudaEventCreate(&fe0); cudaEventCreate(&fe1);
    cudaEventCreate(&me0); cudaEventCreate(&me1);

    auto run_test = [&](auto launch_ffma, const char *name, int blocks, int threads, int iters) {
        // Warmup separately
        for (int i = 0; i < 3; i++) launch_ffma();
        cudaDeviceSynchronize();
        for (int i = 0; i < 3; i++) cudaMemsetAsync(d_buf, 0xab, 4ull*1024*1024*1024, s_m);
        cudaDeviceSynchronize();

        // FFMA alone
        float fa_best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(fe0, s_f);
            launch_ffma();
            cudaEventRecord(fe1, s_f);
            cudaEventSynchronize(fe1);
            float ms; cudaEventElapsedTime(&ms, fe0, fe1);
            if (ms < fa_best) fa_best = ms;
        }

        // Memset alone
        float ma_best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(me0, s_m);
            cudaMemsetAsync(d_buf, 0xab, 4ull*1024*1024*1024, s_m);
            cudaEventRecord(me1, s_m);
            cudaEventSynchronize(me1);
            float ms; cudaEventElapsedTime(&ms, me0, me1);
            if (ms < ma_best) ma_best = ms;
        }

        // Concurrent
        float fc_best = 1e30f, mc_best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(fe0, s_f);
            launch_ffma();
            cudaEventRecord(fe1, s_f);
            cudaEventRecord(me0, s_m);
            cudaMemsetAsync(d_buf, 0xab, 4ull*1024*1024*1024, s_m);
            cudaEventRecord(me1, s_m);
            cudaEventSynchronize(fe1);
            cudaEventSynchronize(me1);
            float fms, mms;
            cudaEventElapsedTime(&fms, fe0, fe1);
            cudaEventElapsedTime(&mms, me0, me1);
            if (fms < fc_best) fc_best = fms;
            if (mms < mc_best) mc_best = mms;
        }

        printf("\n=== %s (%d blocks × %d thr × %d iter) ===\n", name, blocks, threads, iters);
        printf("  FFMA alone:        %.2f ms\n", fa_best);
        printf("  Memset alone:      %.2f ms\n", ma_best);
        printf("  Concurrent:\n");
        printf("    FFMA:            %.2f ms (%.2fx slowdown)\n", fc_best, fc_best/fa_best);
        printf("    Memset:          %.2f ms (%.2fx slowdown)\n", mc_best, mc_best/ma_best);
    };

    // Test 1: max occupancy
    run_test([&]{ ffma_max_occ<<<296, 1024, 0, s_f>>>(d_out, 1000000); },
             "MAX OCCUPANCY (296 × 1024 thr = 2048/SM = 100%)", 296, 1024, 1000000);

    // Test 2: tiny - 1 block, 128 threads (only 1 SM, only 4 warps in it)
    run_test([&]{ ffma_tiny<<<1, 128, 0, s_f>>>(d_out, 10000000); },
             "TINY (1 × 128 thr = uses only 1 SM)", 1, 128, 10000000);

    return 0;
}
