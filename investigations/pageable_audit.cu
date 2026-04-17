// Audit: pageable malloc memory accessed at HBM speed via migration?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void touch_real(volatile float *data, int N, float v, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float x = data[i];
        x = x * 1.0001f + v + seed;
        data[i] = x;
    }
}

int main() {
    cudaSetDevice(0);
    int sm_count = 148;

    auto bench = [&](auto fn, int trials = 5) {
        for (int i = 0; i < 3; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("# AUDIT: Pageable memory access from GPU\n");
    printf("# %-15s %-10s %-12s %-12s %-12s\n", "type", "size_MB", "ms", "BW_GB/s", "first_run_ms");

    for (int sz_mb : {16, 64, 256}) {
        size_t bytes = (size_t)sz_mb * 1024 * 1024;
        size_t N = bytes / sizeof(float);

        // Pageable
        {
            float *p = (float*)aligned_alloc(4096, bytes);
            for (size_t i = 0; i < N; i++) p[i] = 0.0f;

            // FIRST run - cold
            auto t0 = std::chrono::high_resolution_clock::now();
            touch_real<<<sm_count, 256>>>(p, N, 1.0f, sz_mb);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float first_ms = std::chrono::duration<float, std::milli>(t1-t0).count();

            // Subsequent runs - measured "warm"
            float t = bench([&]{ touch_real<<<sm_count, 256>>>(p, N, 1.0f, sz_mb); });
            double rw = 2.0 * bytes;  // read + write
            printf("  %-15s %-10d %-12.3f %-12.1f %-12.2f\n",
                   "pageable", sz_mb, t, rw/(t/1000)/1e9, first_ms);
            free(p);
        }
    }

    // Check if pages were really migrated by querying their attributes
    printf("\n# Verify: cudaPointerGetAttributes on a 'migrated' pageable buffer\n");
    {
        size_t bytes = 64 * 1024 * 1024;
        float *p = (float*)aligned_alloc(4096, bytes);
        memset(p, 0, bytes);

        cudaPointerAttributes attr;
        cudaError_t err = cudaPointerGetAttributes(&attr, p);
        printf("  BEFORE GPU touch: type=%d, device=%d, devPtr=%p, hostPtr=%p (err=%s)\n",
               attr.type, attr.device, attr.devicePointer, attr.hostPointer,
               cudaGetErrorString(err));

        // Touch on GPU
        touch_real<<<sm_count, 256>>>(p, bytes/sizeof(float), 1.0f, 1);
        cudaDeviceSynchronize();

        err = cudaPointerGetAttributes(&attr, p);
        printf("  AFTER GPU touch:  type=%d, device=%d, devPtr=%p, hostPtr=%p (err=%s)\n",
               attr.type, attr.device, attr.devicePointer, attr.hostPointer,
               cudaGetErrorString(err));

        free(p);
    }

    return 0;
}
