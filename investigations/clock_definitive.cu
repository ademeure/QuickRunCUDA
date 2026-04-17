// clock_definitive.cu
// Definitive sustained-clock measurement for B300 SXM6 (sm_103a).
//
// Method:
//   - Run a heavy FFMA workload for ~1 second
//   - Sample %clock64 (SM cycles) and %globaltimer (nanoseconds) at:
//       * kernel start
//       * 25%, 50%, 75%, 100% of the work
//   - Each SM reports its own start/end cycle count and nanosecond count
//   - Average across all SMs to get the sustained clock
//
// Key design choices to make measurement robust:
//   1. Use %globaltimer (wall-clock nanoseconds, GPU-domain) alongside %clock64
//      so we get cycles/ns = clock_GHz without any host timing uncertainty
//   2. Take samples at multiple points (start, 25%, 50%, 75%, end) to detect
//      throttling or boost behavior during the workload
//   3. Each SM independently records its timing via persistent-style blocking
//   4. The FFMA loop is sized to run ~1 second at 2032 MHz; outer loop is
//      #pragma unroll 1 so the compiler cannot LICM it away
//   5. 8 independent FMA chains to saturate the FP32 pipeline
//   6. Verify: globaltimer delta should agree with host-side wall time
//
// Compile:
//   nvcc -arch=sm_103a -O3 -o clock_definitive clock_definitive.cu
//
// Run:
//   ./clock_definitive            (default state)
//   nvidia-smi -lgc 2032          (lock to 2032 MHz)
//   ./clock_definitive            (with lock)
//   nvidia-smi -rgc               (reset)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

// Number of SM-clock samples per SM (start, q1, q2, q3, end)
#define N_SAMPLES 5

// --------------------------------------------------------------------------
// The measurement kernel
//
// Each thread block runs on one SM.
// Thread 0 in the block:
//   - Snapshots (clock64, globaltimer) before and after each quarter of work
//   - Stores the 5 sample pairs into output arrays
// All threads:
//   - Execute the FFMA workload (to keep the SM busy)
// --------------------------------------------------------------------------
__global__ __launch_bounds__(256, 1)
void measure_kernel(
    // Output arrays: [N_SMs][N_SAMPLES] pairs of (cycle, nanosecond)
    unsigned long long* out_cycles,   // [sm_idx * N_SAMPLES + sample]
    unsigned long long* out_nanos,    // [sm_idx * N_SAMPLES + sample]
    int outer_iters,                  // total outer loop iterations
    int seed)                         // DCE-defeat seed
{
    // Each block maps to one SM; use blockIdx.x as SM index.
    int sm_id = blockIdx.x;
    unsigned tid = threadIdx.x;

    // 8 independent FMA chains — same pattern as bench_ffma_peak.cu
    float v0 = __int_as_float(tid + 1) * 1e-30f;
    float v1 = __int_as_float(tid + 2) * 1e-30f;
    float v2 = __int_as_float(tid + 3) * 1e-30f;
    float v3 = __int_as_float(tid + 4) * 1e-30f;
    float v4 = __int_as_float(tid + 5) * 1e-30f;
    float v5 = __int_as_float(tid + 6) * 1e-30f;
    float v6 = __int_as_float(tid + 7) * 1e-30f;
    float v7 = __int_as_float(tid + 8) * 1e-30f;
    float y = 1.5f;  // multiplier != 1.0 to prevent FMA->ADD simplification

    // Quarter boundaries
    int q0 = 0;
    int q1 = outer_iters / 4;
    int q2 = outer_iters / 2;
    int q3 = (3 * outer_iters) / 4;
    int q4 = outer_iters;

    // Macro to snap both counters atomically (they're read sequentially but
    // close enough — only ~1 cycle apart)
#define SNAP(clk, ns) do { \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(clk) : : "memory"); \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns) : : "memory"); \
} while(0)

    unsigned long long clk0, clk1, clk2, clk3, clk4;
    unsigned long long ns0,  ns1,  ns2,  ns3,  ns4;

    // Sync the block so all warps start together before the timer
    __syncthreads();

    // --- Quarter 0 → q1 ---
    if (tid == 0) SNAP(clk0, ns0);
    #pragma unroll 1
    for (int o = q0; o < q1; o++) {
        #pragma unroll
        for (int i = 0; i < 128; i++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
        }
    }
    __syncthreads();

    // --- Quarter q1 → q2 ---
    if (tid == 0) SNAP(clk1, ns1);
    #pragma unroll 1
    for (int o = q1; o < q2; o++) {
        #pragma unroll
        for (int i = 0; i < 128; i++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
        }
    }
    __syncthreads();

    // --- Quarter q2 → q3 ---
    if (tid == 0) SNAP(clk2, ns2);
    #pragma unroll 1
    for (int o = q2; o < q3; o++) {
        #pragma unroll
        for (int i = 0; i < 128; i++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
        }
    }
    __syncthreads();

    // --- Quarter q3 → q4 ---
    if (tid == 0) SNAP(clk3, ns3);
    #pragma unroll 1
    for (int o = q3; o < q4; o++) {
        #pragma unroll
        for (int i = 0; i < 128; i++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
        }
    }
    __syncthreads();
    if (tid == 0) SNAP(clk4, ns4);

    // Defeat DCE: seed-predicated store
    float sum = v0+v1+v2+v3+v4+v5+v6+v7;
    if (__float_as_int(sum) == seed) {
        out_cycles[0] = (unsigned long long)__float_as_int(sum);
    }

    // Thread 0 writes results
    if (tid == 0) {
        int base = sm_id * N_SAMPLES;
        out_cycles[base + 0] = clk0;  out_nanos[base + 0] = ns0;
        out_cycles[base + 1] = clk1;  out_nanos[base + 1] = ns1;
        out_cycles[base + 2] = clk2;  out_nanos[base + 2] = ns2;
        out_cycles[base + 3] = clk3;  out_nanos[base + 3] = ns3;
        out_cycles[base + 4] = clk4;  out_nanos[base + 4] = ns4;
    }
}

int main(int argc, char** argv)
{
    // Select device 0
    CHECK_CUDA(cudaSetDevice(0));

    // Query SM count and clock
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;

    printf("=== B300 Definitive Clock Measurement ===\n");
    printf("Device: %s\n", prop.name);
    printf("SM count: %d\n", sm_count);

    // Query current GPU clock via nvidia-smi (for context)
    printf("\n--- nvidia-smi current SM clock ---\n");
    fflush(stdout);
    system("nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader 2>/dev/null | head -1");
    fflush(stdout);

    // Sizing:
    //   inner loop = 128 FFMAs per chain, 8 chains = 1024 FFMAs per outer iter
    //   At 2032 MHz and 2 FFMAs/cycle (dual-issue) per thread:
    //     cycles per outer iter = 128 / 2 = 64 cycles (with 8 chains in flight)
    //     Actually with latency 4 and 8 chains: throughput = 1 per cycle per chain pair
    //     With pure throughput 2/cyc and 8 chains of 128: 128 outer iters = 128 cycles
    //   We want ~1 second = 2032e6 cycles
    //   outer_iters * 128 cycles = 2032e6  => outer_iters = 15875
    //   Be conservative: use outer_iters = 20000 (enough to run ~1.25s at any freq)
    int outer_iters = 20000;

    // Allocate output buffers
    size_t buf_size = (size_t)sm_count * N_SAMPLES * sizeof(unsigned long long);
    unsigned long long *d_cycles, *d_nanos;
    unsigned long long *h_cycles, *h_nanos;
    CHECK_CUDA(cudaMalloc(&d_cycles, buf_size));
    CHECK_CUDA(cudaMalloc(&d_nanos,  buf_size));
    h_cycles = (unsigned long long*)malloc(buf_size);
    h_nanos  = (unsigned long long*)malloc(buf_size);
    memset(h_cycles, 0, buf_size);
    memset(h_nanos,  0, buf_size);

    // Warmup: 1 short kernel to ensure the GPU is at boost clock
    {
        int wo = 100;
        cudaEvent_t ev_start, ev_stop;
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventRecord(ev_start));
        measure_kernel<<<sm_count, 256>>>(d_cycles, d_nanos, wo, 0);
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));
        float warmup_ms;
        CHECK_CUDA(cudaEventElapsedTime(&warmup_ms, ev_start, ev_stop));
        printf("\nWarmup kernel (outer=%d): %.1f ms\n", wo, warmup_ms);
        CHECK_CUDA(cudaEventDestroy(ev_start));
        CHECK_CUDA(cudaEventDestroy(ev_stop));
    }

    // Main measurement kernel
    printf("\nRunning measurement kernel (outer=%d, ~1-2s at 2032 MHz)...\n", outer_iters);
    fflush(stdout);

    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventRecord(ev_start));
    measure_kernel<<<sm_count, 256>>>(d_cycles, d_nanos, outer_iters, 0);
    CHECK_CUDA(cudaEventRecord(ev_stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    printf("Host-side elapsed (CUDA events): %.1f ms\n", elapsed_ms);

    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_cycles, d_cycles, buf_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_nanos,  d_nanos,  buf_size, cudaMemcpyDeviceToHost));

    // Analysis
    printf("\n--- Per-SM Analysis ---\n");
    printf("%-6s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
           "SM", "start_ns", "end_ns", "dNs", "dCyc", "clk_MHz",
           "q0-q1_MHz");

    double sum_clk_hz = 0.0;
    double sum_q0_hz  = 0.0;  // first quarter (boost into workload)
    double sum_q1_hz  = 0.0;  // second quarter
    double sum_q2_hz  = 0.0;  // third quarter
    double sum_q3_hz  = 0.0;  // fourth quarter (thermal steady state)
    int valid_sms = 0;

    // Track globaltimer drift across SMs to detect cross-SM skew
    unsigned long long first_ns0 = 0, last_ns4 = 0;
    bool first_set = false;

    for (int s = 0; s < sm_count; s++) {
        int base = s * N_SAMPLES;
        unsigned long long c0 = h_cycles[base + 0];
        unsigned long long c1 = h_cycles[base + 1];
        unsigned long long c2 = h_cycles[base + 2];
        unsigned long long c3 = h_cycles[base + 3];
        unsigned long long c4 = h_cycles[base + 4];
        unsigned long long n0 = h_nanos[base + 0];
        unsigned long long n1 = h_nanos[base + 1];
        unsigned long long n2 = h_nanos[base + 2];
        unsigned long long n3 = h_nanos[base + 3];
        unsigned long long n4 = h_nanos[base + 4];

        // Sanity: all timestamps must be non-zero and monotonic
        if (c0 == 0 || n0 == 0 || c4 <= c0 || n4 <= n0) {
            printf("SM %3d: INVALID (c0=%llu n0=%llu c4=%llu n4=%llu)\n",
                   s, c0, n0, c4, n4);
            continue;
        }
        // Monotonicity check within the SM
        if (c1 < c0 || c2 < c1 || c3 < c2 || c4 < c3) {
            printf("SM %3d: NON-MONOTONIC cycles\n", s);
            continue;
        }
        if (n1 < n0 || n2 < n1 || n3 < n2 || n4 < n3) {
            printf("SM %3d: NON-MONOTONIC nanos\n", s);
            continue;
        }

        double total_cyc = (double)(c4 - c0);
        double total_ns  = (double)(n4 - n0);
        double clk_hz = total_cyc / total_ns * 1e9;  // cycles/ns * 1e9 = Hz

        // Per-quarter clocks
        double q0_hz = (double)(c1-c0) / (double)(n1-n0) * 1e9;
        double q1_hz = (double)(c2-c1) / (double)(n2-n1) * 1e9;
        double q2_hz = (double)(c3-c2) / (double)(n3-n2) * 1e9;
        double q3_hz = (double)(c4-c3) / (double)(n4-n3) * 1e9;

        printf("SM %3d  ns0=%12llu  ns4=%12llu  dNs=%9.0f  dCyc=%11.0f  "
               "avg=%.1f MHz  q0=%.1f q1=%.1f q2=%.1f q3=%.1f MHz\n",
               s, n0, n4, total_ns, total_cyc,
               clk_hz/1e6, q0_hz/1e6, q1_hz/1e6, q2_hz/1e6, q3_hz/1e6);

        sum_clk_hz += clk_hz;
        sum_q0_hz  += q0_hz;
        sum_q1_hz  += q1_hz;
        sum_q2_hz  += q2_hz;
        sum_q3_hz  += q3_hz;
        valid_sms++;

        if (!first_set) { first_ns0 = n0; first_set = true; }
        last_ns4 = n4;
    }

    if (valid_sms == 0) {
        fprintf(stderr, "ERROR: No valid SM measurements!\n");
        return 1;
    }

    double avg_clk_mhz = sum_clk_hz / valid_sms / 1e6;
    double avg_q0_mhz  = sum_q0_hz  / valid_sms / 1e6;
    double avg_q1_mhz  = sum_q1_hz  / valid_sms / 1e6;
    double avg_q2_mhz  = sum_q2_hz  / valid_sms / 1e6;
    double avg_q3_mhz  = sum_q3_hz  / valid_sms / 1e6;

    printf("\n=== SUMMARY (%d valid SMs out of %d) ===\n", valid_sms, sm_count);
    printf("Host CUDA-event wall time:        %.1f ms\n", elapsed_ms);
    printf("GPU globaltimer span (first SM):  %.1f ms\n",
           (last_ns4 - first_ns0) / 1e6);
    printf("\nAverage sustained clock across all SMs:\n");
    printf("  Full workload avg:   %.1f MHz\n", avg_clk_mhz);
    printf("  Q0 (0-25%% workload): %.1f MHz\n", avg_q0_mhz);
    printf("  Q1 (25-50%% wkld):   %.1f MHz\n", avg_q1_mhz);
    printf("  Q2 (50-75%% wkld):   %.1f MHz\n", avg_q2_mhz);
    printf("  Q3 (75-100%% wkld):  %.1f MHz\n", avg_q3_mhz);

    // Cross-check: does globaltimer agree with CUDA events?
    // globaltimer ticks at GPU device clock domain, nominally 1 GHz.
    // But on Blackwell, globaltimer is in ACTUAL nanoseconds (wall time).
    double gt_elapsed_ms = (last_ns4 - first_ns0) / 1e6;
    printf("\nCross-check: CUDA events = %.1f ms, globaltimer = %.1f ms  "
           "(ratio %.4f, should be ~1.0)\n",
           elapsed_ms, gt_elapsed_ms, elapsed_ms / gt_elapsed_ms);

    printf("\n--- nvidia-smi SM clock at end ---\n");
    fflush(stdout);
    system("nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader 2>/dev/null | head -1");
    fflush(stdout);

    // Compute TFLOPS implications
    // B300: 148 SMs, 256 CUDA cores/SM (FP32), 2 ops per FFMA
    // FFMA TFLOPS = sm_count * 256_cores * 2_ops * clock_Hz
    int cores_per_sm = 256;  // B300 / Blackwell
    double tflops_avg = (double)sm_count * cores_per_sm * 2.0 * (avg_clk_mhz * 1e6) / 1e12;
    double tflops_1920 = (double)sm_count * cores_per_sm * 2.0 * 1920e6 / 1e12;
    double tflops_2032 = (double)sm_count * cores_per_sm * 2.0 * 2032e6 / 1e12;
    printf("\n--- FFMA TFLOPS Implications (sm=%d, %d cores/SM) ---\n",
           sm_count, cores_per_sm);
    printf("  At measured %.1f MHz: %.2f TFLOPS\n", avg_clk_mhz, tflops_avg);
    printf("  At 1920 MHz:         %.2f TFLOPS (reference)\n", tflops_1920);
    printf("  At 2032 MHz:         %.2f TFLOPS (reference)\n", tflops_2032);
    printf("  Ratio measured/1920: %.4f\n", tflops_avg / tflops_1920);
    printf("  Ratio measured/2032: %.4f\n", tflops_avg / tflops_2032);

    // Cleanup
    cudaFree(d_cycles);
    cudaFree(d_nanos);
    free(h_cycles);
    free(h_nanos);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return 0;
}
