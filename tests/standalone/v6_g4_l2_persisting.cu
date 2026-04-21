// V6 G4: Persistent L2 cache hint via cudaAccessPropertyPersisting
// Pattern: alternating HOT-only and COLD-only kernels
//   Without hint: COLD evicts HOT → next HOT-kernel hits DRAM
//   With hint:    HOT marked Persisting → COLD doesn't evict → next HOT hits L2
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void read_buffer(float* buf, int size_f, int rep) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int r = 0; r < rep; r++) {
        for (int i = gtid; i < size_f; i += total) {
            sum += buf[i];
        }
    }
    if (sum == 1.234567e-30f) buf[gtid] = sum;
}

int main(int argc, char** argv) {
    int mode = (argc > 1) ? atoi(argv[1]) : 0;
    cudaSetDevice(0);

    size_t HOT_BYTES = 16UL * 1024 * 1024;    // 16 MB — fits in 23 MB persist budget
    size_t COLD_BYTES = 256UL * 1024 * 1024;  // 256 MB — defeats L2

    // Bump persisting L2 limit to max
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 79UL * 1024 * 1024);

    float *hot, *cold;
    cudaMalloc(&hot, HOT_BYTES);
    cudaMalloc(&cold, COLD_BYTES);

    // Warmup
    read_buffer<<<1024, 256>>>(hot, HOT_BYTES / 4, 1);
    cudaDeviceSynchronize();
    cudaCtxResetPersistingL2Cache();  // Reset L2 to clean state

    if (mode == 1) {
        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr = hot;
        attr.accessPolicyWindow.num_bytes = HOT_BYTES;  // 16 MB fits in 79 MB persist limit
        attr.accessPolicyWindow.hitRatio = 1.0;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
        printf("Applied Persisting L2 hint to hot buffer (32 MB)\n");
    } else {
        printf("No L2 access hint (default LRU)\n");
    }

    cudaEvent_t es, ee;
    cudaEventCreate(&es); cudaEventCreate(&ee);

    int N_RUNS = 10;
    cudaEventRecord(es);
    for (int i = 0; i < N_RUNS; i++) {
        // Hot kernel (small, should hit L2)
        read_buffer<<<1024, 256>>>(hot, HOT_BYTES / 4, 1);
        // Cold kernel (large, defeats L2)
        read_buffer<<<1024, 256>>>(cold, COLD_BYTES / 4, 1);
    }
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);

    float ms;
    cudaEventElapsedTime(&ms, es, ee);
    double per_iter_ms = ms / N_RUNS;
    double bytes_per_iter = HOT_BYTES + COLD_BYTES;
    double bw_gbps = bytes_per_iter / 1e9 / (per_iter_ms / 1000.0);
    printf("MODE=%d total=%6.2f ms per_iter=%6.2f ms BW=%.1f GB/s\n",
           mode, ms, per_iter_ms, bw_gbps);

    cudaFree(hot); cudaFree(cold);
    return 0;
}
