// Measure cudaEvent timing precision and accuracy
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void busy_kernel(int *out, int iters) {
    float a = 1.0f + threadIdx.x * 0.001f;
    for (int i = 0; i < iters; i++) {
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    }
    if (threadIdx.x == 0) out[blockIdx.x] = (int)a;
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("# B300 cudaEvent timing precision and accuracy\n\n");

    // Test 1: Minimum measurable interval (back-to-back events)
    printf("## Test 1: Empty interval (event0 → event1, no work)\n");
    {
        float min_t = 1e30f, max_t = 0;
        for (int i = 0; i < 100; i++) {
            cudaEventRecord(e0, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < min_t) min_t = ms;
            if (ms > max_t) max_t = ms;
        }
        printf("  Min: %.3f us, Max: %.3f us (over 100 reads)\n", min_t*1000, max_t*1000);
    }

    // Test 2: Resolution (consecutive measurements of same op)
    printf("\n## Test 2: Resolution test - same kernel timed 100x, look for quantization\n");
    {
        // Tiny kernel
        for (int i = 0; i < 5; i++) busy_kernel<<<1, 32, 0, s>>>(d_out, 100);
        cudaDeviceSynchronize();

        float times[100];
        for (int i = 0; i < 100; i++) {
            cudaEventRecord(e0, s);
            busy_kernel<<<1, 32, 0, s>>>(d_out, 100);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&times[i], e0, e1);
        }
        // Find unique values
        float min = 1e30f;
        for (int i = 0; i < 100; i++) if (times[i] < min) min = times[i];
        printf("  Min observed: %.6f us\n", min*1000);

        // Distribution
        printf("  First 10 measurements (us): ");
        for (int i = 0; i < 10; i++) printf("%.3f ", times[i]*1000);
        printf("\n");
    }

    // Test 3: Compare CUDA event vs CPU clock for same workload
    printf("\n## Test 3: CUDA event vs CPU clock comparison (1ms-ish kernel)\n");
    {
        // Warmup
        for (int i = 0; i < 5; i++) busy_kernel<<<148, 128, 0, s>>>(d_out, 50000);
        cudaDeviceSynchronize();

        // CUDA event timing
        cudaEventRecord(e0, s);
        busy_kernel<<<148, 128, 0, s>>>(d_out, 50000);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float cuda_ms; cudaEventElapsedTime(&cuda_ms, e0, e1);

        // CPU timing
        auto t0 = std::chrono::high_resolution_clock::now();
        busy_kernel<<<148, 128, 0, s>>>(d_out, 50000);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        float cpu_ms = std::chrono::duration<float, std::milli>(t1-t0).count();

        printf("  CUDA event:    %.4f ms\n", cuda_ms);
        printf("  CPU std::chrono: %.4f ms\n", cpu_ms);
        printf("  Difference:    %+.4f ms (CPU includes launch + sync overhead)\n", cpu_ms - cuda_ms);
    }

    // Test 4: cudaEventElapsedTime resolution
    printf("\n## Test 4: cudaEventElapsedTime resolution\n");
    {
        // Run kernels of varying iter counts
        for (int iters : {10, 100, 1000, 10000, 100000}) {
            float min_ms = 1e30f;
            for (int trial = 0; trial < 10; trial++) {
                cudaEventRecord(e0, s);
                busy_kernel<<<1, 32, 0, s>>>(d_out, iters);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                if (ms < min_ms) min_ms = ms;
            }
            printf("  iters=%6d: min %.6f ms = %.3f us\n", iters, min_ms, min_ms*1000);
        }
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(d_out);
    return 0;
}
