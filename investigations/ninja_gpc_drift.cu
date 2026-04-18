// B1: inter-GPC clock drift over sustained operation
//
// Hopper/Blackwell expose globaltimer (system-wide) and clock64 (SM clock).
// Question: do different SMs drift in clock64 over sustained operation?
//
// Test: persistent kernel reads clock64 + globaltimer per SM at intervals.
// Compare drift between SMs over time.
//
// Theoretical: if all SMs run from same PLL, no drift. If per-GPC PLLs,
// minor drift is possible.
#include <cuda_runtime.h>
#include <cstdio>
#include <unistd.h>

constexpr int MAX_SAMPLES = 256;

__global__ void measure_drift(unsigned long long *clock_samples,
                              unsigned long long *gt_samples,
                              int n_samples, int spacing_ns) {
    int sm_id;
    asm("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (threadIdx.x != 0) return;
    if (blockIdx.x >= 148) return;
    for (int i = 0; i < n_samples; i++) {
        unsigned long long c, g;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
        clock_samples[blockIdx.x * MAX_SAMPLES + i] = c;
        gt_samples[blockIdx.x * MAX_SAMPLES + i] = g;
        // Spin for spacing_ns nanoseconds (using nanosleep)
        unsigned long long start = c;
        while (true) {
            unsigned long long now;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(now));
            // 2032 MHz = 0.49 ns/cy; 2000 cy = ~1 us
            if (now - start > (unsigned long long)spacing_ns * 2) break;
        }
    }
}

int main(int argc, char**argv) {
    cudaSetDevice(0);
    int n_samples = (argc > 1) ? atoi(argv[1]) : 100;
    int spacing_ns = (argc > 2) ? atoi(argv[2]) : 100000;  // 100 us
    if (n_samples > MAX_SAMPLES) n_samples = MAX_SAMPLES;
    unsigned long long *d_clock; cudaMalloc(&d_clock, 148 * MAX_SAMPLES * sizeof(unsigned long long));
    unsigned long long *d_gt;    cudaMalloc(&d_gt,    148 * MAX_SAMPLES * sizeof(unsigned long long));
    measure_drift<<<148, 32>>>(d_clock, d_gt, n_samples, spacing_ns);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("ERR %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    unsigned long long *h_clock = new unsigned long long[148 * MAX_SAMPLES];
    unsigned long long *h_gt    = new unsigned long long[148 * MAX_SAMPLES];
    cudaMemcpy(h_clock, d_clock, 148 * MAX_SAMPLES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gt,    d_gt,    148 * MAX_SAMPLES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Find min and max clock64 reading at each sample index
    printf("# Inter-GPC clock drift over sustained operation\n");
    printf("# n_samples=%d, spacing=%d ns\n", n_samples, spacing_ns);
    printf("# sample  min_clock  max_clock  range_cy   min_gt(ns)  max_gt(ns)  range_ns\n");
    for (int i = 0; i < n_samples; i += n_samples / 10) {
        if (i >= n_samples) break;
        unsigned long long minc = (unsigned long long)-1, maxc = 0;
        unsigned long long ming = (unsigned long long)-1, maxg = 0;
        for (int b = 0; b < 148; b++) {
            unsigned long long c = h_clock[b * MAX_SAMPLES + i];
            unsigned long long g = h_gt[b * MAX_SAMPLES + i];
            if (c < minc) minc = c;
            if (c > maxc) maxc = c;
            if (g < ming) ming = g;
            if (g > maxg) maxg = g;
        }
        printf("  %5d  %llu  %llu  %llu  %llu  %llu  %llu\n",
               i, minc, maxc, maxc - minc, ming, maxg, maxg - ming);
    }

    // Check final drift between SM 0 and SM 147
    unsigned long long sm0_first = h_clock[0];
    unsigned long long sm0_last  = h_clock[(n_samples-1)];
    unsigned long long sm147_first = h_clock[147 * MAX_SAMPLES];
    unsigned long long sm147_last  = h_clock[147 * MAX_SAMPLES + (n_samples-1)];
    printf("\n# SM 0   delta clock = %llu cy\n", sm0_last - sm0_first);
    printf("# SM 147 delta clock = %llu cy\n", sm147_last - sm147_first);
    printf("# Drift  SM147 vs SM0 over %d samples = %lld cy\n",
           n_samples, (long long)((sm147_last - sm147_first) - (sm0_last - sm0_first)));

    delete[] h_clock; delete[] h_gt;
    return 0;
}
