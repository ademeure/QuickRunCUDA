// AccessPolicyWindow: proper test with workload exceeding L2 capacity
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// Repeated reads of a region - L2 hit rate matters
extern "C" __global__ void rerd(float *in, float *out, int N, int reps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0;
    for (int r = 0; r < reps; r++) {
        for (int i = tid; i < N; i += stride) {
            acc += in[i];
        }
    }
    if (acc == -42.0f) out[tid] = acc;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

    // L2 cache info
    int l2_size, persist_max, window_max;
    cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, 0);
    cudaDeviceGetAttribute(&persist_max, cudaDevAttrMaxPersistingL2CacheSize, 0);
    cudaDeviceGetAttribute(&window_max, cudaDevAttrMaxAccessPolicyWindowSize, 0);
    printf("# B300 L2 cache:\n");
    printf("#   total L2 size: %d bytes (%.1f MB)\n", l2_size, l2_size/(1024.f*1024.f));
    printf("#   max persisting: %d bytes (%.1f MB)\n", persist_max, persist_max/(1024.f*1024.f));
    printf("#   max window size: %d bytes (%.1f MB)\n", window_max, window_max/(1024.f*1024.f));

    // Allocate larger than L2
    size_t bytes = (size_t)512 << 20;  // 512 MB
    int N = bytes / sizeof(float);
    printf("# Test buffer: %zu bytes (%.1f MB), %d floats\n\n", bytes, bytes/(1024.f*1024.f), N);

    float *d_in, *d_out;
    CK(cudaMalloc(&d_in, bytes));
    CK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));
    CK(cudaMemset(d_in, 0x40, bytes));
    CK(cudaMemset(d_out, 0, blocks * threads * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreate(&s));

    auto bench = [&](auto fn, int trials=8) {
        for (int i = 0; i < 2; i++) { fn(); cudaDeviceSynchronize(); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    cudaCtxResetPersistingL2Cache();
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_max);

    int reps = 8;
    int win_sizes_mb[] = {16, 32, 64, 79, 100, 128};
    int n_win = 6;

    printf("# Read window N times, comparing baseline vs persistent\n");
    printf("# %-12s %-12s %-12s %-12s %-12s\n",
           "win_MB", "baseline_ms", "persist_ms", "speedup", "vs_full(MB/s)");

    for (int wi = 0; wi < n_win; wi++) {
        int win_mb = win_sizes_mb[wi];
        size_t win_bytes = (size_t)win_mb << 20;
        if (win_bytes > bytes) win_bytes = bytes;
        int win_N = win_bytes / sizeof(float);

        // Baseline: no policy
        cudaLaunchConfig_t cfg_base = {dim3(blocks),dim3(threads),0,s,nullptr,0};
        float t_base = bench([&]{
            int args_N = win_N, args_reps = reps;
            void *args[] = {&d_in, &d_out, &args_N, &args_reps};
            cudaLaunchKernelExC(&cfg_base, (void*)rerd, args);
        });

        // With persisting policy
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeAccessPolicyWindow;
        attr.val.accessPolicyWindow.base_ptr = d_in;
        attr.val.accessPolicyWindow.num_bytes = (size_t)((win_bytes < window_max) ? win_bytes : window_max);
        attr.val.accessPolicyWindow.hitRatio = 1.0f;
        attr.val.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

        cudaLaunchConfig_t cfg = {dim3(blocks), dim3(threads), 0, s, &attr, 1};

        float t_persist = bench([&]{
            int args_N = win_N, args_reps = reps;
            void *args[] = {&d_in, &d_out, &args_N, &args_reps};
            cudaLaunchKernelExC(&cfg, (void*)rerd, args);
        });

        // Bandwidth: total bytes read = win_bytes * reps
        float bw_base = (win_bytes * reps) / (t_base / 1000.0f) / 1e9f;
        float bw_pers = (win_bytes * reps) / (t_persist / 1000.0f) / 1e9f;
        printf("  %-12d %-12.3f %-12.3f %-12.3f base=%.0f GB/s, pers=%.0f GB/s\n",
               win_mb, t_base, t_persist, t_base / t_persist, bw_base, bw_pers);
    }

    cudaCtxResetPersistingL2Cache();
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
