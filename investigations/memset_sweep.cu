// Sweep FFMA block count to find cudaMemset's actual SM footprint
#include <cuda_runtime.h>
#include <cstdio>
#include <tuple>

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
    float *d_out; cudaMalloc(&d_out, 1024 * 256 * sizeof(float));
    void *d_buf; cudaMalloc(&d_buf, 4ull * 1024 * 1024 * 1024);

    cudaStream_t s_f, s_m;
    cudaStreamCreateWithFlags(&s_f, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_m, cudaStreamNonBlocking);

    cudaEvent_t fe0, fe1, me0, me1;
    cudaEventCreate(&fe0); cudaEventCreate(&fe1);
    cudaEventCreate(&me0); cudaEventCreate(&me1);

    auto bench = [&](int n_blocks, int iters) {
        // FFMA fills n_blocks (one per SM up to 148 means 1/SM at 8 blk/SM cap)
        // With launch_bounds(256, 8), each SM holds 8 blocks max
        // n_blocks up to 1184 (= 8×148) fills SMs FULL

        // Warmup
        for (int i = 0; i < 3; i++) {
            ffma<<<n_blocks, 256, 0, s_f>>>(d_out, iters);
            cudaMemsetAsync(d_buf, 0xab, 4ull*1024*1024*1024, s_m);
        }
        cudaDeviceSynchronize();

        // Memset alone reference
        float ma = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(me0, s_m);
            cudaMemsetAsync(d_buf, 0xab, 4ull*1024*1024*1024, s_m);
            cudaEventRecord(me1, s_m);
            cudaEventSynchronize(me1);
            float ms; cudaEventElapsedTime(&ms, me0, me1);
            if (ms < ma) ma = ms;
        }

        // Concurrent
        float mc = 1e30f, fc = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(fe0, s_f);
            ffma<<<n_blocks, 256, 0, s_f>>>(d_out, iters);
            cudaEventRecord(fe1, s_f);
            cudaEventRecord(me0, s_m);
            cudaMemsetAsync(d_buf, 0xab, 4ull*1024*1024*1024, s_m);
            cudaEventRecord(me1, s_m);
            cudaEventSynchronize(fe1);
            cudaEventSynchronize(me1);
            float fms, mms;
            cudaEventElapsedTime(&fms, fe0, fe1);
            cudaEventElapsedTime(&mms, me0, me1);
            if (fms < fc) fc = fms;
            if (mms < mc) mc = mms;
        }
        return std::make_tuple(ma, fc, mc);
    };

    printf("# Sweep FFMA block count vs cudaMemset slowdown\n");
    printf("# %-12s %-15s %-15s %-15s\n", "ffma_blocks", "memset_alone", "memset_conc", "slowdown");
    int iters = 300000;  // ~3-4 ms FFMA per block
    for (int b : {1, 16, 32, 74, 100, 148, 296, 444, 592, 740, 888, 1036, 1184}) {
        auto [ma, fc, mc] = bench(b, iters);
        printf("  %-12d %-15.2f %-15.2f %-15.2fx\n", b, ma, mc, mc/ma);
    }

    return 0;
}
