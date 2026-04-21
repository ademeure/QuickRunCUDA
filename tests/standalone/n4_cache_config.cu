// N4: cudaCacheConfigPreferShared effect on B300
// Hopper+ unified L1+SMEM 256 KB; carveout configurable via cudaFuncSetCacheConfig
// Test: same kernel doing pointer-chase, different config preferences
#include <cuda_runtime.h>
#include <cstdio>

__global__ void chase(unsigned int* base, int iters, int* time_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned int idx = 0;
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    for (int i = 0; i < iters; i++) idx = base[idx];
    unsigned long long t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (idx == 0xDEADBEEF) base[1] = idx;
    *time_out = (int)((t1 - t0) / iters);
}

int main() {
    cudaSetDevice(0);

    // Setup pointer-chase chain — 1024 lines = 128 KB (fits in default L1)
    int N = 1024;
    int line_size = 128;
    unsigned int* buf;
    cudaMallocManaged(&buf, N * line_size);
    for (int i = 0; i < N; i++) {
        buf[i * (line_size / 4)] = ((i+1) % N) * (line_size / 4);
    }
    cudaDeviceSynchronize();

    int* time_dev;
    cudaMallocManaged(&time_dev, sizeof(int));

    // Test each cache config
    cudaFuncCache configs[] = {
        cudaFuncCachePreferNone,    // default
        cudaFuncCachePreferShared,  // more SMEM, less L1
        cudaFuncCachePreferL1,      // more L1, less SMEM
        cudaFuncCachePreferEqual    // 50/50
    };
    const char* names[] = {"PreferNone", "PreferShared", "PreferL1", "PreferEqual"};

    // Warmup
    chase<<<1, 32>>>(buf, 100, time_dev);
    cudaDeviceSynchronize();

    for (int i = 0; i < 4; i++) {
        cudaFuncSetCacheConfig(chase, configs[i]);
        chase<<<1, 32>>>(buf, 1000, time_dev);
        cudaDeviceSynchronize();
        printf("%s: cy/load = %d\n", names[i], *time_dev);
    }

    return 0;
}
