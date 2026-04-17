// Occupancy as function of register count
#include <cuda_runtime.h>
#include <cstdio>

#define K(N, IDX) \
__launch_bounds__(N) __global__ void kernel_b##N##_##IDX(float *out) { \
    int tid = threadIdx.x + blockIdx.x * blockDim.x; \
    float a[16]; \
    for (int i = 0; i < 16; i++) a[i] = tid + i; \
    for (int i = 0; i < 100; i++) { \
        for (int j = 0; j < 16; j++) a[j] = a[j]*1.0001f + a[(j+1)&15]; \
    } \
    float s = 0; for (int i = 0; i < 16; i++) s += a[i]; \
    if (s < -1e30f) out[tid] = s; \
}

K(32, 1) K(64, 1) K(128, 1) K(256, 1) K(512, 1) K(1024, 1)

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);

    printf("# B300 register count vs occupancy (max blocks/SM)\n");
    printf("# Max regs/SM: %d (Blackwell typical: 65536)\n", prop.regsPerMultiprocessor);
    printf("# Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("# Max threads/block: %d\n\n", prop.maxThreadsPerBlock);

    auto report = [](void *func, int threads, const char *name) {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, func);
        int max_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, func, threads, 0);
        int active_threads = max_blocks * threads;
        float occ = (float)active_threads / 2048.0f * 100;  // 2048 = max threads/SM
        printf("  %-20s regs=%d, smem=%zd, blocks/SM=%d, threads/SM=%d (%.0f%% occ)\n",
               name, attr.numRegs, attr.sharedSizeBytes, max_blocks, active_threads, occ);
    };

    report((void*)kernel_b32_1, 32, "kernel_b32");
    report((void*)kernel_b64_1, 64, "kernel_b64");
    report((void*)kernel_b128_1, 128, "kernel_b128");
    report((void*)kernel_b256_1, 256, "kernel_b256");
    report((void*)kernel_b512_1, 512, "kernel_b512");
    report((void*)kernel_b1024_1, 1024, "kernel_b1024");

    // Direct register-count tests via maxregcount
    printf("\n# Theoretical occupancy table (256 thread blocks):\n");
    int regs[] = {32, 40, 56, 64, 96, 128, 168, 240, 255};
    for (int r : regs) {
        // blocks/SM limited by either reg or thread caps
        int by_regs = 65536 / (r * 256);
        int by_threads = 2048 / 256;  // 8
        int blocks_sm = by_regs < by_threads ? by_regs : by_threads;
        printf("  regs=%3d  blocks/SM=%d  threads/SM=%d  occ=%.0f%%\n",
               r, blocks_sm, blocks_sm * 256, blocks_sm*256/2048.0*100);
    }

    return 0;
}
