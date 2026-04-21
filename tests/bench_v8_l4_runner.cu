// V8 L4 standalone runner: launch warp-trace kernel + analyze variance on host.
// Reports: min, max, mean, stddev, p50, p99 per-warp duration.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <vector>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

#ifndef WORK_CYCLES
#define WORK_CYCLES 1024
#endif

__global__ __launch_bounds__(256, 1)
void trace_kernel(unsigned long long* trace, int n_warps) {
    int warp_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    unsigned long long t_start = 0, t_end = 0;
    if (lane == 0) {
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t_start));
    }

    float a = (float)(threadIdx.x + 1) * 0.001f;
    float b = (float)(threadIdx.x + 2) * 0.001f;
    float c = a;
    #pragma unroll 32
    for (int i = 0; i < WORK_CYCLES; i++) {
        c = a * c + b;
    }

    if (lane == 0) {
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t_end));
        if (warp_id < n_warps) {
            trace[warp_id * 2 + 0] = t_start;
            trace[warp_id * 2 + 1] = t_end;
        }
    }

    if (c == 1.234567e-30f) ((float*)trace)[warp_id * 4] = c;
}

int main(int argc, char** argv) {
    int warps_per_block = 8;   // 256 threads / 32
    int blocks = 148 * 8;      // 148 SMs × 8 blocks/SM = 8 × 8 warps/SM = 64 warps/SM (full occupancy)
    if (argc > 1) blocks = atoi(argv[1]);
    int n_warps = blocks * warps_per_block;

    cudaSetDevice(0);

    unsigned long long* d_trace;
    CK(cudaMalloc(&d_trace, n_warps * 2 * sizeof(unsigned long long)));
    CK(cudaMemset(d_trace, 0, n_warps * 2 * sizeof(unsigned long long)));

    printf("Launching trace_kernel: blocks=%d threads=256 n_warps=%d WORK_CYCLES=%d\n",
           blocks, n_warps, WORK_CYCLES);

    // Warmup
    for (int i = 0; i < 3; i++) {
        trace_kernel<<<blocks, 256>>>(d_trace, n_warps);
    }
    CK(cudaDeviceSynchronize());

    // Timed run
    auto t0 = std::chrono::high_resolution_clock::now();
    trace_kernel<<<blocks, 256>>>(d_trace, n_warps);
    CK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double kernel_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    unsigned long long* h_trace = (unsigned long long*)malloc(n_warps * 2 * sizeof(unsigned long long));
    CK(cudaMemcpy(h_trace, d_trace, n_warps * 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // Compute per-warp durations (in ns, since globaltimer is ns)
    std::vector<double> durations(n_warps);
    unsigned long long global_start = h_trace[0];
    unsigned long long global_end = h_trace[1];
    int valid = 0;
    for (int i = 0; i < n_warps; i++) {
        unsigned long long s = h_trace[i * 2 + 0];
        unsigned long long e = h_trace[i * 2 + 1];
        if (s && e && e > s) {
            durations[valid++] = (double)(e - s);
            if (s < global_start) global_start = s;
            if (e > global_end) global_end = e;
        }
    }
    durations.resize(valid);

    std::sort(durations.begin(), durations.end());
    double mean = 0;
    for (double d : durations) mean += d;
    mean /= valid;
    double variance = 0;
    for (double d : durations) variance += (d - mean) * (d - mean);
    variance /= valid;
    double stddev = sqrt(variance);

    double min_d = durations.front();
    double max_d = durations.back();
    double p50 = durations[valid / 2];
    double p99 = durations[(int)(valid * 0.99)];
    double p999 = durations[(int)(valid * 0.999)];

    printf("\n--- Per-warp duration stats (ns) ---\n");
    printf("  valid warps: %d/%d\n", valid, n_warps);
    printf("  min:   %.0f\n", min_d);
    printf("  p50:   %.0f\n", p50);
    printf("  mean:  %.0f  (stddev %.0f, cv %.2f%%)\n", mean, stddev, stddev / mean * 100);
    printf("  p99:   %.0f\n", p99);
    printf("  p999:  %.0f\n", p999);
    printf("  max:   %.0f  (tail = %.2fx mean)\n", max_d, max_d / mean);
    printf("\n--- Kernel-global wall time ---\n");
    printf("  CUDA wall:  %.1f us\n", kernel_us);
    printf("  global_timer span: %.1f us\n", (double)(global_end - global_start) / 1000.0);
    // Also compute start-skew distribution (separates block-dispatch from per-warp)
    std::vector<double> starts(valid);
    for (int i = 0, j = 0; i < n_warps; i++) {
        unsigned long long s = h_trace[i * 2 + 0];
        unsigned long long e = h_trace[i * 2 + 1];
        if (s && e && e > s) starts[j++] = (double)(s - global_start);
    }
    std::sort(starts.begin(), starts.end());
    double max_start = starts.back();

    printf("\n--- Start-skew (ns from first-started warp) ---\n");
    printf("  min:   %.0f\n", starts.front());
    printf("  p50:   %.0f\n", starts[valid / 2]);
    printf("  p99:   %.0f\n", starts[(int)(valid * 0.99)]);
    printf("  max:   %.0f  (block-dispatch skew)\n", max_start);
    printf("  span = %.1f us (time for all warps to enter work)\n", max_start / 1000);

    printf("\n--- Interpretation ---\n");
    printf("  Per-warp duration CV %.2f%% — includes SM-scheduling variance.\n", stddev / mean * 100);
    printf("  Start-skew span %.1f us — time for 148 SMs to pick up all blocks.\n", max_start/1000);
    printf("  Adjusted 'pure scheduler' variance: duration at %.0f warps.\n", (double)valid);
    printf("  Fairness: min %.0f vs max %.0f → %.2fx spread.\n", min_d, max_d, max_d/min_d);

    free(h_trace);
    cudaFree(d_trace);
    return 0;
}
