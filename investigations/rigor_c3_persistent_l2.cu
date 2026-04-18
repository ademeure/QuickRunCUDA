// C3 v2: simpler L2 persistence test
// HOT region small, sequential chase that hits L2; cold sweep evicts;
// measure throughput before/after with/without persistent.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef HOT_MB
#define HOT_MB 32
#endif
#define HOT_BYTES (HOT_MB * 1024ull * 1024)
#define COLD_BYTES (300ull * 1024 * 1024)

// Bandwidth-style read (8-ILP) of HOT region — measures effective BW
extern "C" __launch_bounds__(256, 8) __global__ void hot_read(unsigned *hot, unsigned *out, int hot_n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    unsigned acc = 0;
    // Each thread reads a portion of hot multiple times to exercise cache reuse
    for (int rep = 0; rep < 4; rep++) {
        for (int i = tid; i < hot_n; i += stride) {
            acc ^= hot[i];
        }
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
    else if (tid == 0) out[0] = acc;
}

extern "C" __global__ void cold_sweep(unsigned *cold, unsigned *out, int cold_n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    unsigned acc = 0;
    for (int i = tid; i < cold_n; i += stride) acc ^= cold[i];
    if (acc == 0xdeadbeef) out[tid] = acc;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("# L2 = %.1f MB, max persisting = %.1f MB, HOT = %d MB, COLD = %.0f MB\n",
           prop.l2CacheSize/1e6, prop.persistingL2CacheMaxSize/1e6,
           HOT_MB, COLD_BYTES/1e6);

    int hot_n = HOT_BYTES / sizeof(unsigned);
    int cold_n = COLD_BYTES / sizeof(unsigned);

    unsigned *d_hot, *d_cold, *d_out;
    cudaMalloc(&d_hot, HOT_BYTES);
    cudaMalloc(&d_cold, COLD_BYTES);
    cudaMalloc(&d_out, 1024 * sizeof(unsigned));
    cudaMemset(d_hot, 0x42, HOT_BYTES);
    cudaMemset(d_cold, 0xab, COLD_BYTES);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize);

    int blocks = 148, threads = 256;

    auto bench = [&](const char* label, int n_warmup, int reps) {
        for (int i = 0; i < n_warmup; i++)
            hot_read<<<blocks, threads>>>(d_hot, d_out, hot_n);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < reps; i++) {
            cudaEventRecord(e0);
            hot_read<<<blocks, threads>>>(d_hot, d_out, hot_n);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        // 4 reps of HOT_BYTES per kernel
        double bytes = HOT_BYTES * 4.0;
        double tbs = bytes / (best/1000) / 1e12;
        printf("  %s: %.4f ms, %.2f TB/s\n", label, best, tbs);
        return tbs;
    };

    printf("\n# === STAGE A: no policy window, fresh ===\n");
    cudaStreamAttrValue empty = {};
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &empty);
    bench("HOT cold       ", 0, 5);
    bench("HOT after warmup", 3, 5);
    cold_sweep<<<blocks, threads>>>(d_cold, d_out, cold_n);
    cudaDeviceSynchronize();
    double normal_after_sweep = bench("HOT after sweep ", 0, 5);

    printf("\n# === STAGE B: persistent L2 on HOT ===\n");
    cudaStreamAttrValue stream_attr = {};
    stream_attr.accessPolicyWindow.base_ptr = d_hot;
    stream_attr.accessPolicyWindow.num_bytes = HOT_BYTES;
    stream_attr.accessPolicyWindow.hitRatio = 1.0f;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attr);

    bench("HOT cold (persist)     ", 0, 5);
    bench("HOT after warmup (persist)", 3, 5);

    cudaStream_t cs;
    cudaStreamCreate(&cs);
    cudaStreamAttrValue cold_attr = {};
    cold_attr.accessPolicyWindow.base_ptr = d_cold;
    cold_attr.accessPolicyWindow.num_bytes = COLD_BYTES;
    cold_attr.accessPolicyWindow.hitRatio = 1.0f;
    cold_attr.accessPolicyWindow.hitProp = cudaAccessPropertyStreaming;
    cold_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(cs, cudaStreamAttributeAccessPolicyWindow, &cold_attr);
    cold_sweep<<<blocks, threads, 0, cs>>>(d_cold, d_out, cold_n);
    cudaStreamSynchronize(cs);

    double persist_after_sweep = bench("HOT after sweep (persist)", 0, 5);

    printf("\n# === Summary ===\n");
    printf("  Without persistent: %.2f TB/s after cold sweep\n", normal_after_sweep);
    printf("  With persistent:    %.2f TB/s after cold sweep\n", persist_after_sweep);
    printf("  Protection ratio:   %.2fx faster\n", persist_after_sweep / normal_after_sweep);

    return 0;
}
