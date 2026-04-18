// B2 v2: L2 persistent — inner-loop reads to amortize launch overhead
//
// Each launch reads WS bytes N_reps times. Total bytes = WS × N_reps.
// If WS fits L2: subsequent reads hit warm L2 → effective BW ≈ L2 peak (~21 TB/s)
// If not: HBM peak (~7 TB/s)
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void k_read_loop(const int4 *p, int *out, size_t N, int reps) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    int4 acc = make_int4(0,0,0,0);
    for (int r = 0; r < reps; r++) {
        for (size_t i = tid; i < N; i += stride) {
            int4 v = p[i];
            acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
        }
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0xDEADBEEF && reps < 0)
        out[threadIdx.x] = acc.x;
}

void test_size(cudaStream_t s, long mb, bool use_persistent) {
    size_t bytes = mb * 1024L * 1024L;
    size_t N_int4 = bytes / 16;
    int4 *d_data; cudaMalloc(&d_data, bytes);
    cudaMemset(d_data, 0, bytes);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    if (use_persistent) {
        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr = (void*)d_data;
        attr.accessPolicyWindow.num_bytes = bytes;
        attr.accessPolicyWindow.hitRatio = 1.0;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
    }

    // Pick reps so each launch ≥ 5 ms (well above launch overhead)
    int reps = (int)(64L * 1024 * 1024 * 1024 / bytes);  // target ~64 GB total per launch
    if (reps < 4) reps = 4;
    if (reps > 1000) reps = 1000;

    // Warmup
    for (int i = 0; i < 2; i++) k_read_loop<<<148*8, 256, 0, s>>>(d_data, d_out, N_int4, reps);
    cudaStreamSynchronize(s);

    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0, s);
        k_read_loop<<<148*8, 256, 0, s>>>(d_data, d_out, N_int4, reps);
        cudaEventRecord(e1, s); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    double tbs = (double)bytes * reps / (best/1000.0) / 1e12;
    double mb_per_launch_total = (double)bytes * reps / 1e6;

    printf("  %s WS=%4ld MB  reps=%4d  tot=%6.0f MB  best=%.3f ms  → %.2f TB/s\n",
           use_persistent ? "persist " : "baseline", mb, reps, mb_per_launch_total, best, tbs);

    if (use_persistent) {
        cudaStreamAttrValue attr = {};
        cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
    }
    cudaFree(d_data); cudaFree(d_out);
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("# B300 L2: %d MB total, persisting cap = %d MB\n",
           (int)(p.l2CacheSize / (1024*1024)), (int)(p.persistingL2CacheMaxSize / (1024*1024)));

    long sizes[] = {1, 8, 32, 64, 79, 100, 128, 256, 512, 1024};
    int n = sizeof(sizes)/sizeof(sizes[0]);

    printf("\n# === BASELINE ===\n");
    for (int i = 0; i < n; i++) test_size(s, sizes[i], false);

    printf("\n# === PERSISTENT (hitRatio=1.0) ===\n");
    for (int i = 0; i < n; i++) test_size(s, sizes[i], true);

    return 0;
}
