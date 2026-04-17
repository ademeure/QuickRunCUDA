// cuFFT performance vs size
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    printf("# B300 cuFFT C2C (1D complex-to-complex) performance\n\n");
    printf("# %-12s %-12s %-12s %-12s\n", "N", "batch", "time_ms", "GFLOPS_est");

    for (int N : {1024, 4096, 16384, 65536, 262144, 1048576}) {
        int batch = std::max(1, 268435456 / N);  // 256 MB worth
        cufftHandle plan;
        cufftPlan1d(&plan, N, CUFFT_C2C, batch);
        cufftSetStream(plan, s);

        size_t bytes = (size_t)N * batch * sizeof(cuFloatComplex);
        cuFloatComplex *d;
        cudaError_t err = cudaMalloc(&d, bytes);
        if (err) { printf("  N=%d: alloc failed\n", N); cufftDestroy(plan); continue; }
        cudaMemset(d, 0, bytes);

        // Warmup
        for (int i = 0; i < 3; i++) cufftExecC2C(plan, d, d, CUFFT_FORWARD);
        cudaStreamSynchronize(s);

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0, s);
            cufftExecC2C(plan, d, d, CUFFT_FORWARD);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        // FFT GFLOPS estimate: 5*N*log2(N)*batch
        double log2N = log2((double)N);
        double flops = 5.0 * N * log2N * batch;
        double gflops = flops / (best/1000.0) / 1e9;
        printf("  %-12d %-12d %-12.3f %-12.0f\n", N, batch, best, gflops);

        cudaFree(d);
        cufftDestroy(plan);
    }

    return 0;
}
