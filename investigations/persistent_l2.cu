// AccessPolicyWindow: persistent L2 caching - measurable benefit?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void read_persistent(const float *hot, const float *cold,
                                            float *out, int N_hot, int N_cold,
                                            int hot_iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0;

    // Read cold, then re-read hot many times
    int cold_per = N_cold / (gridDim.x * blockDim.x);
    for (int i = 0; i < cold_per; i++)
        a += cold[(tid + i*64) & (N_cold-1)];

    for (int it = 0; it < hot_iters; it++) {
        int hot_per = N_hot / (gridDim.x * blockDim.x);
        for (int i = 0; i < hot_per; i++)
            a += hot[(tid + i*64) & (N_hot-1)];
    }

    if (a < -1e30f) out[tid] = a;
}

int main() {
    cudaSetDevice(0);

    int dev = 0;
    int l2_size, max_persist;
    cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, dev);
    cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, dev);
    printf("# B300 L2: total = %d MB, max persisting = %d MB (%.1f%%)\n",
           l2_size/(1024*1024), max_persist/(1024*1024),
           100.0 * max_persist / l2_size);

    // Allocate hot region (small, fits in persisting carveout)
    size_t hot_bytes = 8 * 1024 * 1024;  // 8 MB - well under L2
    size_t cold_bytes = 256 * 1024 * 1024;  // 256 MB cold
    float *d_hot, *d_cold, *d_out;
    cudaMalloc(&d_hot, hot_bytes);
    cudaMalloc(&d_cold, cold_bytes);
    cudaMalloc(&d_out, 1024*1024*sizeof(float));
    cudaMemset(d_hot, 0, hot_bytes);
    cudaMemset(d_cold, 0, cold_bytes);

    int N_hot = hot_bytes / 4;
    int N_cold = cold_bytes / 4;

    cudaStream_t s_normal, s_persist;
    cudaStreamCreate(&s_normal);
    cudaStreamCreate(&s_persist);

    // Set persisting cache window on s_persist
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr = d_hot;
    attr.accessPolicyWindow.num_bytes = hot_bytes;
    attr.accessPolicyWindow.hitRatio = 1.0;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(s_persist, cudaStreamAttributeAccessPolicyWindow, &attr);

    // Configure persisting L2 size to max
    cudaCtxResetPersistingL2Cache();
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, max_persist);

    auto bench = [&](cudaStream_t s, int hot_iters, int trials=50) {
        // Warmup with same kernel - lets persisting L2 prime
        for (int i = 0; i < 5; i++)
            read_persistent<<<148, 256, 0, s>>>(d_hot, d_cold, d_out, N_hot, N_cold, hot_iters);
        cudaStreamSynchronize(s);

        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            read_persistent<<<148, 256, 0, s>>>(d_hot, d_cold, d_out, N_hot, N_cold, hot_iters);
            cudaStreamSynchronize(s);
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("\n# Pattern: read 256 MB cold once, then re-read 8 MB hot N times\n");
    printf("# %-12s %-15s %-15s %-15s\n", "hot_iters", "normal_ms", "persist_ms", "speedup");

    for (int it : {1, 5, 20, 100}) {
        float t_norm = bench(s_normal, it);
        float t_pers = bench(s_persist, it);
        printf("  %-12d %-15.3f %-15.3f %.2fx\n", it, t_norm, t_pers, t_norm/t_pers);
    }

    return 0;
}
