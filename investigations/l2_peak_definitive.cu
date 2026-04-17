// l2_peak_definitive.cu
// Definitive L2 read bandwidth measurement on B300 sm_103a.
//
// Methodology:
//   - 296 CTAs (2 per SM on 148-SM B300), 1024 threads each = 303,104 threads
//   - .cg loads: bypass L1, force L2 as the source (or DRAM if WS > L2)
//   - Unique-per-CTA addresses: each CTA strides through its own slice of the WS
//   - Loop-carried accumulator with XOR to defeat DCE and LICM
//   - Unconditional write of accumulator at end
//   - v4.u32 (128-bit) loads for max memory-level parallelism
//   - ITERS: per-thread loads, chosen by caller to run ~100 ms
//
// Usage (compile directly or via QuickRunCUDA):
//   nvcc -arch=sm_103a -O3 -o l2_peak_definitive l2_peak_definitive.cu
//   OR:
//   ./QuickRunCUDA investigations/l2_peak_definitive.cu -b 296 -t 1024 -T 10 \
//       -0 <ITERS> -2 <WS_BYTES>
//
// WS_BYTES must be a power of 2.

#ifndef NUM_SMS
#define NUM_SMS 148
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#ifndef BLOCKS_PER_SM
#define BLOCKS_PER_SM 2
#endif
#ifndef UNROLL
#define UNROLL 16
#endif

#include <cstdint>
#include <cstdio>
#include <cmath>

// Standalone CUDA benchmark (not using QuickRunCUDA harness)
// because we need precise control over timing and ncu integration.

__global__ __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
void l2_read_cg(const uint32_t* __restrict__ A,
                uint32_t* __restrict__ C,
                int ITERS,
                uint32_t WS_BYTES)
{
    // Each thread strides through the working set with stride = total_threads
    // so the access pattern is fully strided (no CTA-level sharing of cache lines).
    uint32_t tid    = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t nthr   = (uint32_t)(gridDim.x  * blockDim.x);

    // We load 16 bytes (v4.u32) per iteration step; stride in bytes = nthr * 16
    // mask wraps within WS_BYTES (must be power-of-2).
    uint32_t mask   = WS_BYTES - 1u;

    uint32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

    // Base byte offset for this thread: tid * 16 (bytes), then stride by nthr*16 each iter
    uint64_t base_addr = (uint64_t)(uintptr_t)A;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // byte offset: tid*16 + (i+j)*nthr*16, wrapped to WS_BYTES
            uint32_t byte_off = (uint32_t)(
                ((uint64_t)tid * 16ULL +
                 (uint64_t)(i + j) * (uint64_t)nthr * 16ULL)
                & (uint64_t)mask
            );
            // align to 16 bytes (should always be true given above)
            byte_off &= ~15u;
            uint64_t addr = base_addr + byte_off;

            uint32_t x0, x1, x2, x3;
            // .cg = cache global: bypass L1, use L2
            asm volatile(
                "ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                : "l"(addr)
                : "memory"
            );
            acc0 ^= x0; acc1 ^= x1; acc2 ^= x2; acc3 ^= x3;
        }
    }

    // Unconditional write - prevents DCE of the entire loop
    C[tid] = acc0 ^ acc1 ^ acc2 ^ acc3;
}

// Also test .ca (cache all - uses L1) for comparison at small WS
__global__ __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
void l2_read_ca(const uint32_t* __restrict__ A,
                uint32_t* __restrict__ C,
                int ITERS,
                uint32_t WS_BYTES)
{
    uint32_t tid    = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t nthr   = (uint32_t)(gridDim.x  * blockDim.x);
    uint32_t mask   = WS_BYTES - 1u;
    uint64_t base_addr = (uint64_t)(uintptr_t)A;

    uint32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            uint32_t byte_off = (uint32_t)(
                ((uint64_t)tid * 16ULL +
                 (uint64_t)(i + j) * (uint64_t)nthr * 16ULL)
                & (uint64_t)mask
            );
            byte_off &= ~15u;
            uint64_t addr = base_addr + byte_off;

            uint32_t x0, x1, x2, x3;
            asm volatile(
                "ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                : "l"(addr)
                : "memory"
            );
            acc0 ^= x0; acc1 ^= x1; acc2 ^= x2; acc3 ^= x3;
        }
    }

    C[tid] = acc0 ^ acc1 ^ acc2 ^ acc3;
}

#include <cuda_runtime.h>
#include <cstring>

#define CHECK(call) do {                                        \
    cudaError_t _e = (call);                                   \
    if (_e != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(_e));   \
        exit(1);                                               \
    }                                                          \
} while(0)

struct BenchResult {
    double bw_tb_s;   // effective bandwidth in TB/s (16 bytes per load × loads)
    double ms;        // kernel time in ms
};

static BenchResult run_bench(
    const char* label,
    bool use_ca,
    const uint32_t* d_A,
    uint32_t* d_C,
    int num_blocks,
    int block_size,
    int ITERS,
    size_t WS_BYTES,
    int warmup,
    int trials
) {
    cudaEvent_t ev_start, ev_stop;
    CHECK(cudaEventCreate(&ev_start));
    CHECK(cudaEventCreate(&ev_stop));

    uint32_t ws32 = (uint32_t)WS_BYTES;
    dim3 grid(num_blocks), block(block_size);

    // Warmup
    for (int w = 0; w < warmup; w++) {
        if (use_ca)
            l2_read_ca<<<grid, block>>>(d_A, d_C, ITERS, ws32);
        else
            l2_read_cg<<<grid, block>>>(d_A, d_C, ITERS, ws32);
    }
    CHECK(cudaDeviceSynchronize());

    // Timed trials
    double best_bw = 0.0;
    double total_ms = 0.0;
    for (int t = 0; t < trials; t++) {
        CHECK(cudaEventRecord(ev_start));
        if (use_ca)
            l2_read_ca<<<grid, block>>>(d_A, d_C, ITERS, ws32);
        else
            l2_read_cg<<<grid, block>>>(d_A, d_C, ITERS, ws32);
        CHECK(cudaEventRecord(ev_stop));
        CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));

        // Total bytes = num_threads × ITERS × 16 bytes/load
        long long num_threads = (long long)num_blocks * block_size;
        double bytes = (double)num_threads * (double)ITERS * 16.0;
        double bw = bytes / (ms * 1e-3) / 1e12;  // TB/s

        total_ms += ms;
        if (bw > best_bw) best_bw = bw;
    }

    CHECK(cudaEventDestroy(ev_start));
    CHECK(cudaEventDestroy(ev_stop));

    BenchResult r;
    r.bw_tb_s = best_bw;
    r.ms = total_ms / trials;
    return r;
}

int main(int argc, char** argv) {
    // Select GPU 0
    CHECK(cudaSetDevice(0));

    // Query SM count
    int sm_count = 0;
    CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    printf("GPU has %d SMs\n", sm_count);

    int num_blocks = sm_count * BLOCKS_PER_SM;
    int block_size = BLOCK_SIZE;
    long long num_threads = (long long)num_blocks * block_size;

    printf("Launch config: %d blocks × %d threads = %lld threads\n",
           num_blocks, block_size, num_threads);
    printf("UNROLL = %d, BLOCKS_PER_SM = %d\n\n", UNROLL, BLOCKS_PER_SM);

    // Allocate buffer large enough for largest working set (256 MB + C buffer)
    // Buffer A: 256 MB for data, Buffer C: num_threads × 4 bytes
    size_t max_ws = 256ULL * 1024 * 1024;  // 256 MB
    size_t c_size = (size_t)num_threads * 4;

    uint32_t* d_A;
    uint32_t* d_C;
    CHECK(cudaMalloc(&d_A, max_ws));
    CHECK(cudaMalloc(&d_C, c_size));
    CHECK(cudaMemset(d_A, 0, max_ws));
    CHECK(cudaMemset(d_C, 0, c_size));

    // Working set sizes to test (powers of 2)
    struct WsConfig {
        const char* label;
        size_t ws_bytes;
        const char* tier;
    };

    WsConfig configs[] = {
        { "1 MB",   1ULL << 20,  "L1+L2 (should be L1-dominated)" },
        { "4 MB",   4ULL << 20,  "L2 only (small, near-side dominant)" },
        { "16 MB",  16ULL << 20, "L2 only (comfortably fits)" },
        { "32 MB",  32ULL << 20, "L2 only (one partition ~60 MB)" },
        { "64 MB",  64ULL << 20, "L2 plateau (both partitions)" },
        { "128 MB", 128ULL << 20,"L2 at capacity (126.5 MB)" },
        { "256 MB", 256ULL << 20,"L2 overflow → DRAM" },
    };
    int nconfigs = (int)(sizeof(configs) / sizeof(configs[0]));

    // For each WS, choose ITERS so kernel runs ~100ms:
    // target_bytes = 100e-3 s × (assume 20 TB/s) = 2 TB
    // ITERS = 2 TB / (num_threads × 16 B)
    double target_bytes = 2e12;  // 2 TB to read
    int base_iters = (int)(target_bytes / ((double)num_threads * 16.0));
    // Round up to multiple of UNROLL
    base_iters = ((base_iters + UNROLL - 1) / UNROLL) * UNROLL;
    if (base_iters < UNROLL) base_iters = UNROLL;

    printf("Targeting ~%.0f TB total data read per run\n", target_bytes / 1e12);
    printf("ITERS per thread = %d (= %.1f GB per thread)\n\n",
           base_iters, (double)base_iters * 16.0 / 1e9);

    printf("%-12s  %-8s  %-50s  %12s  %12s  %12s  %8s\n",
           "Working Set", "Hint", "Tier", "BW .cg (TB/s)", "BW .ca (TB/s)", "Best (TB/s)", "Time(ms)");
    printf("%-12s  %-8s  %-50s  %12s  %12s  %12s  %8s\n",
           "------------", "--------", "--------------------------------------------------",
           "------------", "------------", "------------", "--------");

    for (int i = 0; i < nconfigs; i++) {
        size_t ws = configs[i].ws_bytes;
        if (ws > max_ws) {
            printf("%-12s  SKIP (> buffer size)\n", configs[i].label);
            continue;
        }

        // Flush L2 by touching a different region (write then re-zero)
        // We use cudaMemset to flush any L2 residency
        CHECK(cudaMemset(d_A, 0xAB, ws));
        CHECK(cudaDeviceSynchronize());

        BenchResult res_cg = run_bench(
            "cg", false,
            d_A, d_C, num_blocks, block_size,
            base_iters, ws, 2, 5
        );

        CHECK(cudaMemset(d_A, 0xCD, ws));
        CHECK(cudaDeviceSynchronize());

        BenchResult res_ca = run_bench(
            "ca", true,
            d_A, d_C, num_blocks, block_size,
            base_iters, ws, 2, 5
        );

        double best = (res_cg.bw_tb_s > res_ca.bw_tb_s) ? res_cg.bw_tb_s : res_ca.bw_tb_s;

        printf("%-12s  .cg/.ca  %-50s  %12.2f  %12.2f  %12.2f  %8.1f\n",
               configs[i].label,
               configs[i].tier,
               res_cg.bw_tb_s, res_ca.bw_tb_s, best, res_cg.ms);
    }

    // Also sweep ITERS sensitivity at 32 MB (the "canonical" L2 WS)
    printf("\n--- ITERS sweep at 32 MB (canonical L2 working set) ---\n");
    size_t ws32 = 32ULL * 1024 * 1024;
    CHECK(cudaMemset(d_A, 0, ws32));

    int iters_list[] = {
        UNROLL, UNROLL*4, UNROLL*16, UNROLL*64, UNROLL*256
    };
    int niters = (int)(sizeof(iters_list) / sizeof(iters_list[0]));
    printf("%-12s  %12s  %8s\n", "ITERS", "BW .cg (TB/s)", "Time(ms)");
    printf("%-12s  %12s  %8s\n", "------------", "------------", "--------");
    for (int k = 0; k < niters; k++) {
        int iters = iters_list[k];
        BenchResult r = run_bench(
            "cg", false,
            d_A, d_C, num_blocks, block_size,
            iters, ws32, 2, 5
        );
        printf("%-12d  %12.2f  %8.1f\n", iters, r.bw_tb_s, r.ms);
    }

    // Thread count sweep at 32 MB — to understand TLP dependence
    printf("\n--- Thread count sweep at 32 MB (checking TLP saturation) ---\n");
    printf("%-12s  %-12s  %12s  %8s\n", "Blocks", "Threads/SM", "BW .cg (TB/s)", "Time(ms)");
    printf("%-12s  %-12s  %12s  %8s\n", "------------", "------------", "------------", "--------");

    struct TLPConfig { int blocks; int bs; };
    TLPConfig tlp_configs[] = {
        { sm_count,              128  },   // 128 thr/SM (1 CTA × 128)
        { sm_count,              256  },   // 256 thr/SM (1 CTA × 256)
        { sm_count,              512  },   // 512 thr/SM (1 CTA × 512)
        { sm_count,              1024 },   // 1024 thr/SM (1 CTA × 1024)
        { sm_count * 2,          512  },   // 1024 thr/SM (2 CTA × 512)
        { sm_count * 2,          1024 },   // 2048 thr/SM (2 CTA × 1024) — max for bs=1024
    };
    int ntlp = (int)(sizeof(tlp_configs) / sizeof(tlp_configs[0]));
    for (int k = 0; k < ntlp; k++) {
        int nb = tlp_configs[k].blocks;
        int bs = tlp_configs[k].bs;
        long long nt = (long long)nb * bs;
        int iters_k = ((int)(target_bytes / ((double)nt * 16.0)) + UNROLL - 1) / UNROLL * UNROLL;
        if (iters_k < UNROLL) iters_k = UNROLL;

        // Reallocate C if needed (different thread count)
        if ((size_t)nt * 4 > c_size) {
            CHECK(cudaFree(d_C));
            c_size = (size_t)nt * 4;
            CHECK(cudaMalloc(&d_C, c_size));
        }

        BenchResult r = run_bench(
            "cg", false,
            d_A, d_C, nb, bs,
            iters_k, ws32, 2, 5
        );
        double thr_per_sm = (double)(nb * bs) / sm_count;
        printf("%-12d  %-12.0f  %12.2f  %8.1f\n", nb, thr_per_sm, r.bw_tb_s, r.ms);
    }

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    printf("\nDone.\n");
    return 0;
}
