// V52: Dual-issue clean retest using V8's 128-deep recipe.
//
// Background:
//   V49 (8-deep + #pragma unroll 1 + 128 thr × launch_bounds(128,2))
//     measured FFMA+LOP3 dual-issue at 55%.
//   V50 (warp-specialized, same depth) measured 74%.
//   SASS verify (corrections/SASS_VERIFY_DUAL_ISSUE.md) found V49's inner body
//     was 8 FFMA + 8 LOP3 + UIADD3 + UISETP + BRA.U — the loop overhead
//     contaminates the dual-issue measurement.
//   V8 reaches 97.64% pure FFMA SoL with 256 thr × launch_bounds(256,1) +
//     128-deep inner unroll (16 outer × 8-way ILP).
//
// Goal: re-measure dual-issue with V8's recipe so the inner body has
//   ≥128 FFMA + ≥128 LOP3 per outer iter, with only 1 BRA.
//
// Tests (per geometry):
//   - solo FFMA  (V8 baseline)
//   - solo LOP3
//   - dual FFMA+LOP3 interleaved (one FFMA + one LOP3 per slot)
//
// Geometries:
//   - 1 CTA / SM × 256 thr  (= 2 warps/SMSP, V8's recipe)
//   - 2 CTA / SM × 256 thr  (= 4 warps/SMSP)
//
// ILP sweep: 4 / 8 / 16
//
// Anti-DCE: store final XOR-reduced accumulator to global under threadIdx.x==0.
// Anti-LICM: registers init from threadIdx.x.
// Both clock64 and cudaEvent timing.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <vector>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// ---------------------------------------------------------------------------
// Kernel: parameterised by op-mode, ILP, blocks-per-SM, and outer iter count.
// Inner unroll is 16 (V8 pattern), so total inner ops per outer iter
// = 16 × ILP × ops_per_slot. With ILP=8, that's 128 of each op type per outer.
// ---------------------------------------------------------------------------
template<int MODE, int ILP, int BLOCKS_PER_SM, int N_OUTER>
__global__ __launch_bounds__(256, BLOCKS_PER_SM)
void v52_kernel(unsigned long long* clk_out, unsigned* sink) {
    int tid = threadIdx.x;

    // Anti-LICM: depend on tid so compiler can't precompute.
    float f[16];
    unsigned u[16];
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        f[k] = (float)(tid + k * 3);
        u[k] = (unsigned)(tid * 7u + k * 13u);
    }
    // Distinct register holding 1.5f for FFMA's 2nd source (V8 pattern).
    float bc = (float)(tid * 2 + 1) * 0.0f + 1.5f; // depends on tid, equals 1.5f at runtime

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Outer loop: #pragma unroll 1 forces real branch. Inner: unroll 16 × ILP.
    #pragma unroll 1
    for (int it = 0; it < N_OUTER; it++) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            if (MODE == 0) {
                // Solo FFMA — V8 recipe: fma %0, %0, %1, %0 (distinct register source)
                #pragma unroll
                for (int k = 0; k < ILP; k++) {
                    asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(f[k]) : "f"(bc));
                }
            } else if (MODE == 1) {
                // Solo LOP3
                #pragma unroll
                for (int k = 0; k < ILP; k++) {
                    asm volatile("lop3.b32 %0, %0, 0xa5a5a5a5, 0x12345678, 0x96;" : "+r"(u[k]));
                }
            } else { // MODE == 2: dual FFMA+LOP3 interleaved per slot
                #pragma unroll
                for (int k = 0; k < ILP; k++) {
                    asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(f[k]) : "f"(bc));
                    asm volatile("lop3.b32 %0, %0, 0xa5a5a5a5, 0x12345678, 0x96;" : "+r"(u[k]));
                }
            }
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    // Anti-DCE: unconditional XOR reduction stored under (tid==0 || conditional).
    // Use unconditional store of acc XOR clock-diff so compiler cannot DCE.
    unsigned acc = 0;
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        acc ^= u[k] ^ __float_as_uint(f[k]);
    }
    // Store cycle delta per block (always); store sink so DCE can't kill ops.
    if (tid == 0) {
        clk_out[blockIdx.x] = t1 - t0;
    }
    // Always-execute store of acc (predicated to keep results live but avoid memory blowup).
    if (tid == (acc & 0)) {  // (acc & 0) == 0 always; predicate is dynamic so compiler can't fold
        sink[blockIdx.x * blockDim.x + tid] = acc;
    }
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

struct RunResult {
    float wall_ms;
    double clk_cycles_med;   // median per-CTA clock64 delta
    double total_ops;        // total scalar ops issued (one mode-op = 1 unit)
    double glane_per_s;      // ops / time / 1e9
};

template<int MODE, int ILP, int BLOCKS_PER_SM, int N_OUTER>
RunResult run_one(int blocks, int threads, int n_runs,
                  unsigned long long* d_clk, unsigned* d_sink,
                  cudaEvent_t e0, cudaEvent_t e1)
{
    // Warmup
    v52_kernel<MODE, ILP, BLOCKS_PER_SM, N_OUTER><<<blocks, threads>>>(d_clk, d_sink);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<float> ms_samples;
    for (int r = 0; r < n_runs; r++) {
        cudaEventRecord(e0);
        v52_kernel<MODE, ILP, BLOCKS_PER_SM, N_OUTER><<<blocks, threads>>>(d_clk, d_sink);
        cudaEventRecord(e1);
        CK(cudaEventSynchronize(e1));
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        ms_samples.push_back(ms);
    }
    std::sort(ms_samples.begin(), ms_samples.end());
    float ms_med = ms_samples[ms_samples.size() / 2];

    // Per-CTA clock cycles (median across CTAs)
    std::vector<unsigned long long> clk_host(blocks);
    CK(cudaMemcpy(clk_host.data(), d_clk, blocks * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost));
    std::sort(clk_host.begin(), clk_host.end());
    double clk_med = (double)clk_host[blocks / 2];

    // Ops counting:
    //   per-thread inner ops per outer iter = 16 * ILP * (1 if MODE<2 else 2)
    //   total threads = blocks * threads
    //   total ops = per_thread * N_OUTER * total_threads
    int ops_per_slot = (MODE == 2) ? 2 : 1;
    double per_thread_ops = (double)16 * ILP * ops_per_slot * N_OUTER;
    double total_ops = per_thread_ops * (double)blocks * (double)threads;
    double glane = total_ops / (ms_med / 1e3) / 1e9;

    RunResult out{ms_med, clk_med, total_ops, glane};
    return out;
}

int main(int argc, char** argv) {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;

    printf("=== V52 Dual-issue CLEAN retest (V8 recipe: 128-deep, lb(256,1)) ===\n");
    int clk_khz = 0;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
    printf("Device: %s, SMs=%d, clock_max=%d MHz\n",
           prop.name, sm_count, clk_khz / 1000);

    // Allocate buffers sized for the largest BLOCKS we'll use (148 * 2 = 296)
    const int MAX_BLOCKS = sm_count * 2;
    unsigned long long* d_clk;
    unsigned* d_sink;
    CK(cudaMalloc(&d_clk, MAX_BLOCKS * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_sink, MAX_BLOCKS * 256 * sizeof(unsigned)));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_RUNS = 5;            // median of 5
    const int THREADS = 256;         // V8 recipe
    const int N_OUTER_ILP4  = 4096;  // ~ tune for >= 5 ms
    const int N_OUTER_ILP8  = 2048;
    const int N_OUTER_ILP16 = 1024;

    // For each (BPS, ILP), run solo FFMA, solo LOP3, dual.
    auto report = [&](const char* tag, RunResult solo_f, RunResult solo_l, RunResult dual) {
        // dual.total_ops counts FFMA+LOP3 separately, so dual.glane is the sum-rate.
        // For "dual_FFMA_only" + "dual_LOP3_only" we know they're equal in the kernel
        // (one each per slot), so each is dual.glane / 2.
        double dual_each = dual.glane_per_s / 2.0;
        double max_solo = std::max(solo_f.glane_per_s, solo_l.glane_per_s);
        double sum_solo = solo_f.glane_per_s + solo_l.glane_per_s;
        double eff_vs_max = dual.glane_per_s / max_solo * 100.0;
        double eff_vs_sum = dual.glane_per_s / sum_solo * 100.0;
        printf("\n--- %s ---\n", tag);
        printf("  solo FFMA  : ms=%7.3f  Glane/s=%8.2f  (clk_med=%.0f)\n",
               solo_f.wall_ms, solo_f.glane_per_s, solo_f.clk_cycles_med);
        printf("  solo LOP3  : ms=%7.3f  Glane/s=%8.2f  (clk_med=%.0f)\n",
               solo_l.wall_ms, solo_l.glane_per_s, solo_l.clk_cycles_med);
        printf("  dual FF+LP : ms=%7.3f  Glane/s=%8.2f total  (=%.2f each)  (clk_med=%.0f)\n",
               dual.wall_ms, dual.glane_per_s, dual_each, dual.clk_cycles_med);
        printf("  dual / max(solo) = %6.2f%%   (>120%% => pipes overlap)\n", eff_vs_max);
        printf("  dual / sum(solo) = %6.2f%%   (~100%% => fully separate pipes)\n", eff_vs_sum);
        // FFMA peak references
        // 148 SMs × 128 FP32 cores × 2 = 37888 FFMA-lane-issues per cycle
        // at 2032 MHz boost = 37888 * 2.032e9 = 76.97 GFFMA-lane/s = 76970 Glane/s
        double ffma_peak = 148.0 * 128.0 * 2.032; // Glane/s
        printf("  solo FFMA / FFMA_peak(@2032) = %6.2f%%\n", solo_f.glane_per_s / ffma_peak * 100.0);
    };

    // ----------------------------------------------------------------
    // Geometry A: 1 CTA / SM (V8 recipe)
    // ----------------------------------------------------------------
    {
        const int BPS = 1;
        const int blocks = sm_count * BPS; // 148
        printf("\n========== Geometry A: %d blocks × %d threads (BPS=%d, 2 warps/SMSP) ==========\n",
               blocks, THREADS, BPS);

        // ILP=4
        {
            auto sf = run_one<0, 4, 1, N_OUTER_ILP4>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            auto sl = run_one<1, 4, 1, N_OUTER_ILP4>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            auto du = run_one<2, 4, 1, N_OUTER_ILP4>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            report("Geom A, ILP=4", sf, sl, du);
        }
        // ILP=8 (V8's recipe exactly)
        {
            auto sf = run_one<0, 8, 1, N_OUTER_ILP8>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            auto sl = run_one<1, 8, 1, N_OUTER_ILP8>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            auto du = run_one<2, 8, 1, N_OUTER_ILP8>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            report("Geom A, ILP=8", sf, sl, du);
        }
        // ILP=16
        {
            auto sf = run_one<0, 16, 1, N_OUTER_ILP16>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            auto sl = run_one<1, 16, 1, N_OUTER_ILP16>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            auto du = run_one<2, 16, 1, N_OUTER_ILP16>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            report("Geom A, ILP=16", sf, sl, du);
        }
    }

    // ----------------------------------------------------------------
    // Geometry B: 2 CTAs / SM (4 warps/SMSP)
    // ----------------------------------------------------------------
    {
        const int BPS = 2;
        const int blocks = sm_count * BPS; // 296
        printf("\n========== Geometry B: %d blocks × %d threads (BPS=%d, 4 warps/SMSP) ==========\n",
               blocks, THREADS, BPS);

        // ILP=4
        {
            auto sf = run_one<0, 4, 2, N_OUTER_ILP4>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            auto sl = run_one<1, 4, 2, N_OUTER_ILP4>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            auto du = run_one<2, 4, 2, N_OUTER_ILP4>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            report("Geom B, ILP=4", sf, sl, du);
        }
        // ILP=8
        {
            auto sf = run_one<0, 8, 2, N_OUTER_ILP8>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            auto sl = run_one<1, 8, 2, N_OUTER_ILP8>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            auto du = run_one<2, 8, 2, N_OUTER_ILP8>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            report("Geom B, ILP=8", sf, sl, du);
        }
        // ILP=16
        {
            auto sf = run_one<0, 16, 2, N_OUTER_ILP16>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            auto sl = run_one<1, 16, 2, N_OUTER_ILP16>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            auto du = run_one<2, 16, 2, N_OUTER_ILP16>(blocks, THREADS, N_RUNS, d_clk, d_sink, e0, e1);
            report("Geom B, ILP=16", sf, sl, du);
        }
    }

    printf("\nReference peaks (B300 SXM6 sm_103a, 148 SMs, 2032 MHz boost):\n");
    printf("  FP32 FFMA: 76.97 TFLOPS = 38485 Glane/s (V8 hits 97.64%% = 37582 Glane/s)\n");
    printf("  LOP3 (ALU pipe): same theoretical 1 inst/SMSP/cy = same 38485 Glane/s\n");

    cudaFree(d_clk);
    cudaFree(d_sink);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 0;
}
