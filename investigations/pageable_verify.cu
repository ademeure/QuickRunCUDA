// Verify pageable memory access bandwidth - the prior test gave 444 GB/s which is impossible
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cstring>

extern "C" __global__ void touch_real(volatile float *data, int N, float v, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float x = data[i];
        x = x * 1.0001f + v + seed;  // can't be optimized to constant
        data[i] = x;
    }
}

int main() {
    cudaSetDevice(0);

    auto bench = [&](auto fn, int trials = 5) {
        for (int i = 0; i < 2; i++) fn();
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

    int sm_count = 148;
    printf("# B300 GPU access to host memory - careful BW measurement\n\n");
    printf("# Pattern: data[i] = data[i]*1.0001 + v + seed (read+write, anti-DCE)\n");
    printf("# %-15s %-15s %-15s %-15s %-15s\n",
           "type", "size_MB", "time_ms", "RW_bytes_GB", "BW_GB/s");

    for (int sz_mb : {16, 64, 256}) {
        size_t bytes = (size_t)sz_mb * 1024 * 1024;
        size_t N = bytes / sizeof(float);

        // 1. Pageable malloc
        {
            float *p = (float*)aligned_alloc(4096, bytes);
            for (size_t i = 0; i < N; i++) p[i] = 0.0f;  // commit pages

            int seed = sz_mb;
            float t = bench([&]{ touch_real<<<sm_count, 256>>>(p, N, 1.0f, seed); });
            // Read + write = 2 * bytes
            double rw = 2.0 * bytes / 1e9;
            printf("  %-15s %-15d %-15.3f %-15.3f %-15.1f\n",
                   "pageable", sz_mb, t, rw, rw/(t/1000));
            free(p);
        }

        // 2. Pinned (cudaMallocHost)
        {
            float *p;
            cudaMallocHost(&p, bytes);
            for (size_t i = 0; i < N; i++) p[i] = 0.0f;

            int seed = sz_mb + 1000;
            float t = bench([&]{ touch_real<<<sm_count, 256>>>(p, N, 1.0f, seed); });
            double rw = 2.0 * bytes / 1e9;
            printf("  %-15s %-15d %-15.3f %-15.3f %-15.1f\n",
                   "pinned", sz_mb, t, rw, rw/(t/1000));
            cudaFreeHost(p);
        }

        // 3. Device memory (for reference - max BW)
        {
            float *p;
            cudaMalloc(&p, bytes);
            cudaMemset(p, 0, bytes);

            int seed = sz_mb + 2000;
            float t = bench([&]{ touch_real<<<sm_count, 256>>>(p, N, 1.0f, seed); });
            double rw = 2.0 * bytes / 1e9;
            printf("  %-15s %-15d %-15.3f %-15.3f %-15.1f\n",
                   "device", sz_mb, t, rw, rw/(t/1000));
            cudaFree(p);
        }
    }

    return 0;
}
