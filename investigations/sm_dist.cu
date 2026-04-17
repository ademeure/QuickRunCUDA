// Block-to-SM distribution: how does the GPU load-balance work across SMs?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>

__device__ uint32_t get_smid() {
    uint32_t sm; asm("mov.u32 %0, %%smid;" : "=r"(sm));
    return sm;
}

extern "C" __global__ void record_smid(uint32_t *smids, int *counts, int n_blocks) {
    if (threadIdx.x == 0) {
        uint32_t sm = get_smid();
        if (blockIdx.x < n_blocks) smids[blockIdx.x] = sm;
        atomicAdd(&counts[sm], 1);
    }
}

extern "C" __global__ void busy_with_smid(uint32_t *smids, int n_blocks, int iters) {
    if (threadIdx.x == 0 && blockIdx.x < n_blocks) {
        smids[blockIdx.x] = get_smid();
    }
    float a = (float)threadIdx.x;
    for (int i = 0; i < iters; i++) a = a*1.0001f + 0.0001f;
    if (a < -1e30f) smids[gridDim.x] = (int)a;
}

int main() {
    cudaSetDevice(0);

    const int N_SM = 200;  // generous
    int *d_counts;
    cudaMalloc(&d_counts, N_SM * sizeof(int));

    auto run_dist = [&](int n_blocks, int threads_per_block) {
        cudaMemset(d_counts, 0, N_SM * sizeof(int));
        uint32_t *d_smids; cudaMalloc(&d_smids, n_blocks * sizeof(uint32_t));
        record_smid<<<n_blocks, threads_per_block>>>(d_smids, d_counts, n_blocks);
        cudaDeviceSynchronize();

        int counts[N_SM];
        cudaMemcpy(counts, d_counts, N_SM * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_smids);

        int active = 0, max_c = 0, min_c = INT_MAX, total = 0;
        for (int i = 0; i < N_SM; i++) {
            if (counts[i] > 0) {
                active++;
                total += counts[i];
                if (counts[i] > max_c) max_c = counts[i];
                if (counts[i] < min_c) min_c = counts[i];
            }
        }
        if (active == 0) min_c = 0;

        printf("  %-12d %-12d %-12d %-12d %-12d\n",
               n_blocks, active, min_c, max_c, total);
    };

    printf("# B300 block-to-SM scheduling distribution\n");
    printf("# (counts blocks scheduled on each SM)\n\n");
    printf("# %-12s %-12s %-12s %-12s %-12s\n",
           "n_blocks", "SMs_active", "min/SM", "max/SM", "total");

    for (int n : {1, 4, 16, 32, 64, 128, 148, 200, 296, 444, 1000, 5000}) {
        run_dist(n, 32);
    }

    // Now test: does the scheduler favor low SM IDs first?
    printf("\n# Block 0..147 SM IDs (148 blocks - one per SM ideally):\n");
    {
        int n = 148;
        uint32_t *d_smids; cudaMalloc(&d_smids, n * sizeof(uint32_t));
        cudaMemset(d_counts, 0, N_SM * sizeof(int));
        record_smid<<<n, 32>>>(d_smids, d_counts, n);
        cudaDeviceSynchronize();
        uint32_t smids[200];
        cudaMemcpy(smids, d_smids, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        printf("  ");
        for (int i = 0; i < std::min(n, 30); i++) printf("%d ", smids[i]);
        printf(" ... last 5: ");
        for (int i = n-5; i < n; i++) printf("%d ", smids[i]);
        printf("\n");
        cudaFree(d_smids);
    }

    // Long-running kernel, check that 296 blocks really do get scheduled
    printf("\n# 296 blocks (expecting 2/SM) with iters\n");
    {
        int n = 296;
        uint32_t *d_smids; cudaMalloc(&d_smids, n * sizeof(uint32_t));
        cudaMemset(d_counts, 0, N_SM * sizeof(int));
        busy_with_smid<<<n, 32>>>(d_smids, n, 1000);
        cudaDeviceSynchronize();

        int counts[N_SM]; cudaMemcpy(counts, d_counts, N_SM*sizeof(int), cudaMemcpyDeviceToHost);
        // Recount from smids since busy_with_smid doesn't write counts
        uint32_t smids[300];
        cudaMemcpy(smids, d_smids, n*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        memset(counts, 0, sizeof(counts));
        for (int i = 0; i < n; i++) counts[smids[i]]++;

        int active = 0, max_c = 0, min_c = INT_MAX;
        for (int i = 0; i < N_SM; i++) {
            if (counts[i] > 0) { active++; if (counts[i] > max_c) max_c = counts[i]; if (counts[i] < min_c) min_c = counts[i]; }
        }
        printf("  %d blocks: %d SMs active, min=%d, max=%d/SM\n", n, active, min_c, max_c);
        cudaFree(d_smids);
    }

    return 0;
}
