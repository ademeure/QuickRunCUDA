// launch_latency_sweep.cu
// Characterize CUDA kernel launch latency vs grid size on B300 sm_103a.
//
// Tests:
//   1. GPU-side kernel time (cudaEvent pair bracketing the kernel)
//   2. CPU-side launch time (clock_gettime around cudaLaunchKernel)
//   3. Variants: kernel<<<>>>, cudaLaunchKernelEx, cudaGraphLaunch
//   4. Pinned vs unpinned context (not the kernel, just the surroundings)
//   5. Sweep: blocks = 1,10,100,1K,10K,100K,1M; threads = 1,32,128,256,512,1024
//
// Compile: nvcc -arch=sm_103a -O3 -o launch_latency_sweep investigations/launch_latency_sweep.cu
// Run:     ./launch_latency_sweep

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <math.h>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Truly empty kernel — compiler should emit minimal code
__global__ void noop_kernel() {}

// Minimal kernel that does one global write (forces real scheduling)
__global__ void minimal_kernel(int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = 1;
}

// ---------------------------------------------------------------------------
// Helper: get nanoseconds (CPU wall clock)
// ---------------------------------------------------------------------------
static inline int64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// ---------------------------------------------------------------------------
// Structures to hold results
// ---------------------------------------------------------------------------
struct Stats {
    double mean_us;
    double median_us;
    double p5_us;
    double p95_us;
    double min_us;
    double max_us;
};

Stats compute_stats(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    Stats s;
    s.min_us = v.front();
    s.max_us = v.back();
    s.mean_us = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    s.median_us = v[v.size() / 2];
    s.p5_us  = v[(int)(v.size() * 0.05)];
    s.p95_us = v[(int)(v.size() * 0.95)];
    return s;
}

// ---------------------------------------------------------------------------
// Print table row
// ---------------------------------------------------------------------------
void print_row(const char* label, int blocks, int threads,
               const char* variant,
               const Stats& gpu, const Stats& cpu) {
    printf("| %-28s | %7d | %6d | %7.2f | %7.2f | %7.2f | %7.2f | %7.2f | %7.2f |\n",
           label,
           blocks, threads,
           gpu.median_us, gpu.p5_us, gpu.p95_us,
           cpu.median_us, cpu.p5_us, cpu.p95_us);
}

// ---------------------------------------------------------------------------
// Benchmark: GPU event time for kernel<<<grid,block>>>
// ---------------------------------------------------------------------------
Stats bench_gpu_triple_chevron(dim3 grid, dim3 block, cudaStream_t stream,
                                int warmup, int iters) {
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        noop_kernel<<<grid, block, 0, stream>>>();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples(iters);
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(ev0, stream));
        noop_kernel<<<grid, block, 0, stream>>>();
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        samples[i] = (double)ms * 1000.0; // us
    }

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return compute_stats(samples);
}

// ---------------------------------------------------------------------------
// Benchmark: CPU-side launch time (clock around cudaLaunchKernel, async)
// No sync inside loop — measures just the driver call cost.
// ---------------------------------------------------------------------------
Stats bench_cpu_launch_time(dim3 grid, dim3 block, cudaStream_t stream,
                             int warmup, int iters) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        noop_kernel<<<grid, block, 0, stream>>>();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples(iters);
    void* args[] = {};
    for (int i = 0; i < iters; i++) {
        // CPU time of just the launch call
        int64_t t0 = now_ns();
        cudaLaunchKernel((const void*)noop_kernel, grid, block, args, 0, stream);
        int64_t t1 = now_ns();
        samples[i] = (double)(t1 - t0) / 1000.0; // us
        // Let GPU pipeline drain occasionally to avoid stream overflow
        if ((i & 63) == 63) CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return compute_stats(samples);
}

// ---------------------------------------------------------------------------
// Benchmark: cudaLaunchKernelEx (with attrs struct — same kernel)
// ---------------------------------------------------------------------------
Stats bench_gpu_launch_kernel_ex(dim3 grid, dim3 block, cudaStream_t stream,
                                  int warmup, int iters) {
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim  = grid;
    cfg.blockDim = block;
    cfg.stream   = stream;
    cfg.attrs    = nullptr;
    cfg.numAttrs = 0;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, noop_kernel));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples(iters);
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(ev0, stream));
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, noop_kernel));
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        samples[i] = (double)ms * 1000.0; // us
    }

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return compute_stats(samples);
}

// ---------------------------------------------------------------------------
// Benchmark: CUDA Graph (pre-instantiated, repeated launch)
// ---------------------------------------------------------------------------
Stats bench_gpu_graph(dim3 grid, dim3 block, cudaStream_t stream,
                       int warmup, int iters) {
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // Build the graph once
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    noop_kernel<<<grid, block, 0, stream>>>();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples(iters);
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(ev0, stream));
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        samples[i] = (double)ms * 1000.0; // us
    }

    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return compute_stats(samples);
}

// ---------------------------------------------------------------------------
// Benchmark: CPU-side graph launch time
// ---------------------------------------------------------------------------
Stats bench_cpu_graph_time(dim3 grid, dim3 block, cudaStream_t stream,
                            int warmup, int iters) {
    // Build the graph once
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    noop_kernel<<<grid, block, 0, stream>>>();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples(iters);
    for (int i = 0; i < iters; i++) {
        int64_t t0 = now_ns();
        cudaGraphLaunch(graphExec, stream);
        int64_t t1 = now_ns();
        samples[i] = (double)(t1 - t0) / 1000.0; // us
        if ((i & 63) == 63) CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return compute_stats(samples);
}

// ---------------------------------------------------------------------------
// Grid-size scaling: GPU time only, triple-chevron, many block counts
// ---------------------------------------------------------------------------
void run_grid_scaling_sweep(cudaStream_t stream) {
    const int warmup = 50;
    const int iters  = 500;

    int thread_configs[] = {1, 32, 128, 256, 512, 1024};
    int block_configs[]  = {1, 10, 100, 1000, 10000, 100000, 1000000};

    printf("\n");
    printf("=== GPU EVENT TIME vs GRID SIZE (triple-chevron, noop kernel) ===\n");
    printf("| %-28s | %7s | %6s | %7s | %7s | %7s |\n",
           "config", "blocks", "threads", "med(us)", "p5(us)", "p95(us)");
    printf("|%-30s|%9s|%8s|%9s|%9s|%9s|\n",
           "------------------------------", "---------", "--------",
           "---------", "---------", "---------");

    for (int ti = 0; ti < (int)(sizeof(thread_configs)/sizeof(thread_configs[0])); ti++) {
        int threads = thread_configs[ti];
        for (int bi = 0; bi < (int)(sizeof(block_configs)/sizeof(block_configs[0])); bi++) {
            int blocks = block_configs[bi];
            // Skip configs that exceed device limits
            if ((int64_t)blocks * threads > 2LL * 1024 * 1024 * 1024) continue;

            dim3 grid(blocks, 1, 1);
            dim3 block(threads, 1, 1);

            // For very large grids, use fewer iters to avoid very long runs
            int actual_iters = (blocks >= 100000) ? 100 : iters;
            int actual_warmup = (blocks >= 100000) ? 10 : warmup;

            Stats gpu = bench_gpu_triple_chevron(grid, block, stream,
                                                  actual_warmup, actual_iters);

            char label[64];
            snprintf(label, sizeof(label), "noop %d×%d", blocks, threads);
            printf("| %-28s | %7d | %6d | %7.2f | %7.2f | %7.2f |\n",
                   label, blocks, threads,
                   gpu.median_us, gpu.p5_us, gpu.p95_us);
            fflush(stdout);
        }
        printf("|%-30s|%9s|%8s|%9s|%9s|%9s|\n",
               "------------------------------", "---------", "--------",
               "---------", "---------", "---------");
    }
}

// ---------------------------------------------------------------------------
// Variant sweep: for a few representative grid sizes, compare all variants
// ---------------------------------------------------------------------------
void run_variant_sweep(cudaStream_t stream) {
    const int warmup = 50;
    const int iters  = 500;

    struct Config { int blocks; int threads; };
    Config configs[] = {
        {1,    1},
        {1,    32},
        {1,    1024},
        {32,   32},
        {148,  128},    // 1 CTA/SM (B300 = 148 SMs)
        {296,  1024},   // 2 CTA/SM, full blocks
        {1000, 256},
        {10000, 128},
        {100000, 32},
    };

    printf("\n");
    printf("=== GPU EVENT TIME: VARIANT COMPARISON (GPU time, all in µs) ===\n");
    printf("| %-20s | %7s | %6s | %7s | %7s | %7s | %7s |\n",
           "variant", "blocks", "threads", "med", "p5", "p95", "min");
    printf("|%-22s|%9s|%8s|%9s|%9s|%9s|%9s|\n",
           "----------------------", "---------", "--------",
           "---------", "---------", "---------", "---------");

    for (int ci = 0; ci < (int)(sizeof(configs)/sizeof(configs[0])); ci++) {
        int blocks  = configs[ci].blocks;
        int threads = configs[ci].threads;
        dim3 grid(blocks);
        dim3 block(threads);

        int actual_iters  = (blocks >= 10000) ? 100 : iters;
        int actual_warmup = (blocks >= 10000) ? 10  : warmup;

        Stats g_chev = bench_gpu_triple_chevron(grid, block, stream, actual_warmup, actual_iters);
        Stats g_ex   = bench_gpu_launch_kernel_ex(grid, block, stream, actual_warmup, actual_iters);
        Stats g_gr   = bench_gpu_graph(grid, block, stream, actual_warmup, actual_iters);

        char label[64];
        snprintf(label, sizeof(label), "chevron(%d,%d)", blocks, threads);
        printf("| %-20s | %7d | %6d | %7.2f | %7.2f | %7.2f | %7.2f |\n",
               label, blocks, threads, g_chev.median_us, g_chev.p5_us, g_chev.p95_us, g_chev.min_us);
        snprintf(label, sizeof(label), "LaunchKernelEx(%d,%d)", blocks, threads);
        printf("| %-20s | %7d | %6d | %7.2f | %7.2f | %7.2f | %7.2f |\n",
               label, blocks, threads, g_ex.median_us, g_ex.p5_us, g_ex.p95_us, g_ex.min_us);
        snprintf(label, sizeof(label), "GraphLaunch(%d,%d)", blocks, threads);
        printf("| %-20s | %7d | %6d | %7.2f | %7.2f | %7.2f | %7.2f |\n",
               label, blocks, threads, g_gr.median_us, g_gr.p5_us, g_gr.p95_us, g_gr.min_us);
        printf("|%-22s|%9s|%8s|%9s|%9s|%9s|%9s|\n",
               "----------------------", "---------", "--------",
               "---------", "---------", "---------", "---------");
        fflush(stdout);
    }
}

// ---------------------------------------------------------------------------
// CPU-side launch call time (just the driver enqueue, no sync)
// ---------------------------------------------------------------------------
void run_cpu_side_sweep(cudaStream_t stream) {
    const int warmup = 50;
    const int iters  = 500;

    struct Config { int blocks; int threads; };
    Config configs[] = {
        {1,    1},
        {1,    1024},
        {148,  1024},
        {1000, 256},
        {10000, 128},
        {100000, 32},
        {1000000, 1},
    };

    printf("\n");
    printf("=== CPU-SIDE LAUNCH CALL TIME (clock_gettime, no sync, µs) ===\n");
    printf("| %-24s | %7s | %6s | %7s | %7s | %7s | %7s |\n",
           "variant", "blocks", "threads", "med", "p5", "p95", "min");
    printf("|%-26s|%9s|%8s|%9s|%9s|%9s|%9s|\n",
           "--------------------------", "---------", "--------",
           "---------", "---------", "---------", "---------");

    for (int ci = 0; ci < (int)(sizeof(configs)/sizeof(configs[0])); ci++) {
        int blocks  = configs[ci].blocks;
        int threads = configs[ci].threads;
        dim3 grid(blocks);
        dim3 block(threads);

        int actual_iters  = (blocks >= 100000) ? 200 : iters;
        int actual_warmup = (blocks >= 100000) ? 20  : warmup;

        Stats c_chev = bench_cpu_launch_time(grid, block, stream, actual_warmup, actual_iters);
        Stats c_gr   = bench_cpu_graph_time(grid, block, stream, actual_warmup, actual_iters);

        char label[64];
        snprintf(label, sizeof(label), "cudaLaunchKernel(%d,%d)", blocks, threads);
        printf("| %-24s | %7d | %6d | %7.3f | %7.3f | %7.3f | %7.3f |\n",
               label, blocks, threads, c_chev.median_us, c_chev.p5_us, c_chev.p95_us, c_chev.min_us);
        snprintf(label, sizeof(label), "cudaGraphLaunch(%d,%d)", blocks, threads);
        printf("| %-24s | %7d | %6d | %7.3f | %7.3f | %7.3f | %7.3f |\n",
               label, blocks, threads, c_gr.median_us, c_gr.p5_us, c_gr.p95_us, c_gr.min_us);
        printf("|%-26s|%9s|%8s|%9s|%9s|%9s|%9s|\n",
               "--------------------------", "---------", "--------",
               "---------", "---------", "---------", "---------");
        fflush(stdout);
    }
}

// ---------------------------------------------------------------------------
// Fine-grained: many streams (does parallelism reduce per-launch cost?)
// ---------------------------------------------------------------------------
void run_multistream_test() {
    const int warmup = 20;
    const int iters  = 200;

    int stream_counts[] = {1, 2, 4, 8, 16};
    dim3 grid(148), block(128);  // 1 CTA/SM

    printf("\n");
    printf("=== MULTI-STREAM: GPU TIME PER LAUNCH (148×128, concurrent) ===\n");
    printf("| %7s | %7s | %7s | %7s | %7s |\n",
           "streams", "med(us)", "p5(us)", "p95(us)", "total_us");
    printf("|%9s|%9s|%9s|%9s|%9s|\n",
           "---------", "---------", "---------", "---------", "---------");

    for (int si = 0; si < (int)(sizeof(stream_counts)/sizeof(stream_counts[0])); si++) {
        int nstreams = stream_counts[si];

        std::vector<cudaStream_t> streams(nstreams);
        std::vector<cudaEvent_t> ev0s(nstreams), ev1s(nstreams);
        for (int i = 0; i < nstreams; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaEventCreate(&ev0s[i]));
            CUDA_CHECK(cudaEventCreate(&ev1s[i]));
        }

        // Warmup
        for (int w = 0; w < warmup; w++) {
            for (int i = 0; i < nstreams; i++) {
                noop_kernel<<<grid, block, 0, streams[i]>>>();
            }
            for (int i = 0; i < nstreams; i++) {
                CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            }
        }

        // Measure: all streams launch simultaneously, wait for last
        std::vector<double> per_kernel_samples(iters);
        for (int it = 0; it < iters; it++) {
            for (int i = 0; i < nstreams; i++) {
                CUDA_CHECK(cudaEventRecord(ev0s[i], streams[i]));
                noop_kernel<<<grid, block, 0, streams[i]>>>();
                CUDA_CHECK(cudaEventRecord(ev1s[i], streams[i]));
            }
            float max_ms = 0.0f;
            for (int i = 0; i < nstreams; i++) {
                CUDA_CHECK(cudaEventSynchronize(ev1s[i]));
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev0s[i], ev1s[i]));
                if (ms > max_ms) max_ms = ms;
            }
            per_kernel_samples[it] = (double)max_ms * 1000.0;
        }

        Stats s = compute_stats(per_kernel_samples);
        printf("| %7d | %7.2f | %7.2f | %7.2f | %7.2f |\n",
               nstreams, s.median_us, s.p5_us, s.p95_us, s.median_us * nstreams);

        for (int i = 0; i < nstreams; i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
            CUDA_CHECK(cudaEventDestroy(ev0s[i]));
            CUDA_CHECK(cudaEventDestroy(ev1s[i]));
        }
        fflush(stdout);
    }
}

// ---------------------------------------------------------------------------
// Sanity: measure event overhead itself
// ---------------------------------------------------------------------------
void run_event_overhead_test(cudaStream_t stream) {
    printf("\n");
    printf("=== EVENT OVERHEAD (no kernel, just record+elapsed) ===\n");

    const int iters = 500;
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // Warmup
    for (int i = 0; i < 50; i++) {
        CUDA_CHECK(cudaEventRecord(ev0, stream));
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));
    }

    std::vector<double> samples(iters);
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(ev0, stream));
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        samples[i] = (double)ms * 1000.0;
    }

    Stats s = compute_stats(samples);
    printf("event-only (no kernel): median=%.3f us  p5=%.3f  p95=%.3f  min=%.3f\n",
           s.median_us, s.p5_us, s.p95_us, s.min_us);
    printf("(This is the floor: measured GPU kernel time = kernel + event resolution floor)\n");

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
}

// ---------------------------------------------------------------------------
// Very large grid sweep to stress-test linearity claim
// ---------------------------------------------------------------------------
void run_large_grid_sweep(cudaStream_t stream) {
    printf("\n");
    printf("=== VERY LARGE GRID SWEEP (GPU event time, noop, 32 threads/block) ===\n");
    printf("| %10s | %10s | %7s | %7s | %7s |\n",
           "blocks", "total_thr", "med(us)", "p5(us)", "p95(us)");
    printf("|%12s|%12s|%9s|%9s|%9s|\n",
           "------------", "------------", "---------", "---------", "---------");

    int64_t block_counts[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                               1024, 2048, 4096, 8192, 16384, 32768,
                               65536, 131072, 262144, 524288, 1048576};
    int threads = 32;

    for (int bi = 0; bi < (int)(sizeof(block_counts)/sizeof(block_counts[0])); bi++) {
        int64_t blocks = block_counts[bi];
        dim3 grid((int)blocks);
        dim3 block(threads);

        int actual_iters  = (blocks >= 100000) ? 50 : (blocks >= 10000 ? 100 : 300);
        int actual_warmup = (blocks >= 100000) ? 5  : (blocks >= 10000 ? 10  : 20);

        Stats s = bench_gpu_triple_chevron(grid, block, stream, actual_warmup, actual_iters);

        printf("| %10lld | %10lld | %7.2f | %7.2f | %7.2f |\n",
               (long long)blocks, (long long)(blocks * threads),
               s.median_us, s.p5_us, s.p95_us);
        fflush(stdout);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Compute cap: %d.%d\n", prop.major, prop.minor);
    // clockRate removed in CUDA 13+ for sm_103 devices
    printf("\n");

    // Create a default stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Trigger CUDA context init
    noop_kernel<<<1, 1, 0, stream>>>();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    run_event_overhead_test(stream);
    run_grid_scaling_sweep(stream);
    run_large_grid_sweep(stream);
    run_variant_sweep(stream);
    run_cpu_side_sweep(stream);
    run_multistream_test();

    CUDA_CHECK(cudaStreamDestroy(stream));
    printf("\nDone.\n");
    return 0;
}
