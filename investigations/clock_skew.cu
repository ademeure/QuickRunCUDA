// Measure if %clock64 across SMs starts simultaneously or has skew
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void gather_clocks(unsigned long long *out, unsigned int *smids) {
    if (threadIdx.x == 0) {
        unsigned long long c, g;
        unsigned int smid;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        out[blockIdx.x * 2 + 0] = c;      // SM's clock64 at kernel start
        out[blockIdx.x * 2 + 1] = g;      // GPU wall-clock at kernel start
        smids[blockIdx.x] = smid;
    }
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm = prop.multiProcessorCount;

    unsigned long long *d_out;
    unsigned int *d_smids;
    cudaMalloc(&d_out, sm * 2 * sizeof(unsigned long long));
    cudaMalloc(&d_smids, sm * sizeof(unsigned int));

    // Launch 1 block per SM, measure clocks
    gather_clocks<<<sm, 32>>>(d_out, d_smids);
    cudaDeviceSynchronize();

    unsigned long long h_out[256 * 2];
    unsigned int h_smids[256];
    cudaMemcpy(h_out, d_out, sm * 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_smids, d_smids, sm * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Find min globaltimer (= kernel start) and compute relative offsets
    unsigned long long g_min = h_out[1];
    unsigned long long c_min_at_g_min = h_out[0];
    int min_idx = 0;
    for (int i = 1; i < sm; i++) {
        if (h_out[i * 2 + 1] < g_min) {
            g_min = h_out[i * 2 + 1];
            c_min_at_g_min = h_out[i * 2];
            min_idx = i;
        }
    }

    printf("# B300 SM clock skew test (1 block per SM)\n");
    printf("# Min globaltimer: %llu ns (block %d on SM %u)\n", g_min, min_idx, h_smids[min_idx]);
    printf("#\n");

    // For each SM, compute: (clock64 - c_min_at_g_min) - (globaltimer - g_min) * 2.032
    // This shows if SM clocks are synchronized
    printf("# Clock drift analysis (assuming 2.032 GHz):\n");
    printf("# %-6s %-6s %-15s %-15s %-12s\n",
           "blk", "smid", "globaltimer_Δ", "clock64_Δ", "drift_cy");

    // Print first 20 and any SMs with unusual drift
    int max_drift = 0;
    int max_drift_smid = -1;
    for (int i = 0; i < sm; i++) {
        long long g_delta = (long long)h_out[i * 2 + 1] - (long long)g_min;
        long long c_delta = (long long)h_out[i * 2] - (long long)c_min_at_g_min;
        long long expected_c_delta = (long long)(g_delta * 2.032);
        long long drift = c_delta - expected_c_delta;
        if (i < 10 || abs(drift) > 10000) {
            printf("  %-6d %-6u %-15lld %-15lld %-12lld\n",
                   i, h_smids[i], g_delta, c_delta, drift);
        }
        if (abs(drift) > max_drift) {
            max_drift = abs(drift);
            max_drift_smid = h_smids[i];
        }
    }

    printf("\n# Max drift: %d cycles on SM %d\n", max_drift, max_drift_smid);
    printf("# (Small drift = SM clocks running in sync; large drift = GPC boot phase difference)\n");

    cudaFree(d_out); cudaFree(d_smids);
    return 0;
}
