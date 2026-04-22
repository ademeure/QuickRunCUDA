// V54: membar.{cta,gl,sys} per-fence cost via N-issue scaling.
//
// Settles UNRESOLVED #3 from corrections/RETEST_PROPOSALS.md:
//   __threadfence_system catalog spread:
//     08_sync_primitives:  1750 cy / 861 ns
//     V9_THREADFENCE_COST: 3042 cy / 1486 ns
//     DSMEM_REFERENCE:     2870 cy
//   1.74x discrepancy.
//
// Method:
//   Single CTA, single warp, single thread (lane 0 only).
//   For each scope in {cta, gl, sys}:
//     Sweep N in {0,1,2,4,8,16,32} fences in sequence between two clock64.
//     Measure 21 trials per cell, take median.
//   Linear fit total_cy = const + per_fence * N.
//   per_fence is the per-fence steady-state cost (pipelined or serial).
//   const is fixed loop + clock read overhead, including the N=0 baseline.
//
// Anti-reordering / liveness:
//   - One atom.global.add anchor BEFORE clock64_start: forces a real
//     prior global write so the fence has something to order.
//   - One atom.global.add anchor AFTER clock64_end: ensures the fence
//     is observed on the side of memory ordering (the compiler cannot
//     remove a fence whose ordering effect is still visible).
//   - asm volatile + "memory" clobber on every fence and every clock64.
//
// SASS verify expected:
//   N=0:  zero MEMBAR
//   N=1:  one MEMBAR.<scope>
//   N=k:  exactly k MEMBAR.<scope> instructions
//   plus 2 ATOMG (atomic.add anchors) and 2 CS2R clock64.
//
// Compile:
//   nvcc -arch=sm_103a -O3 -std=c++17 v54_membar_isolation.cu -o /tmp/v54
// SASS:
//   cuobjdump --dump-sass /tmp/v54 | grep -E "MEMBAR|ATOM|CS2R"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <vector>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// SCOPE: 0=cta, 1=gl(GPU), 2=sys
template<int SCOPE, int N_ISSUE>
__global__ __launch_bounds__(32, 1)
void v54_membar(unsigned long long* out, unsigned* dummy_global) {
    if (threadIdx.x != 0) return;

    // Anchor BEFORE: force a real global write that the fence will order.
    unsigned anchor_in;
    asm volatile("atom.global.add.u32 %0, [%1], 1;"
        : "=r"(anchor_in) : "l"(dummy_global) : "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll
    for (int i = 0; i < N_ISSUE; i++) {
        if      (SCOPE == 0) asm volatile("membar.cta;" ::: "memory");
        else if (SCOPE == 1) asm volatile("membar.gl;"  ::: "memory");
        else                 asm volatile("membar.sys;" ::: "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    // Anchor AFTER: another atomic so fence's ordering effect is observable
    // (the compiler cannot DCE a fence whose post-anchor depends on prior writes).
    unsigned anchor_out;
    asm volatile("atom.global.add.u32 %0, [%1], 1;"
        : "=r"(anchor_out) : "l"(dummy_global + 1) : "memory");

    // Write delta out, also leak anchors so they aren't DCE'd.
    out[0] = t1 - t0;
    out[1] = (unsigned long long)anchor_in + (unsigned long long)anchor_out;
}

template<int SCOPE>
unsigned long long run_one(int n_issue, unsigned long long* d_out, unsigned* d_dummy, int trials) {
    auto launch = [&](int ni) {
        switch (ni) {
            case  0: v54_membar<SCOPE, 0><<<1,32>>>(d_out, d_dummy); break;
            case  1: v54_membar<SCOPE, 1><<<1,32>>>(d_out, d_dummy); break;
            case  2: v54_membar<SCOPE, 2><<<1,32>>>(d_out, d_dummy); break;
            case  4: v54_membar<SCOPE, 4><<<1,32>>>(d_out, d_dummy); break;
            case  8: v54_membar<SCOPE, 8><<<1,32>>>(d_out, d_dummy); break;
            case 16: v54_membar<SCOPE,16><<<1,32>>>(d_out, d_dummy); break;
            case 32: v54_membar<SCOPE,32><<<1,32>>>(d_out, d_dummy); break;
            default: fprintf(stderr,"bad N=%d\n",ni); exit(1);
        }
    };
    // Warmup
    for (int w = 0; w < 3; w++) { launch(n_issue); cudaDeviceSynchronize(); }
    // Trials
    std::vector<unsigned long long> samples(trials);
    for (int s = 0; s < trials; s++) {
        launch(n_issue);
        cudaDeviceSynchronize();
        cudaMemcpy(&samples[s], d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    }
    std::sort(samples.begin(), samples.end());
    return samples[trials / 2];  // median
}

// Linear fit y = a + b*x via least-squares; returns (a, b, R^2).
struct Fit { double a, b, r2; };
Fit linfit(const std::vector<int>& xs, const std::vector<unsigned long long>& ys) {
    int n = (int)xs.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) { sx += xs[i]; sy += (double)ys[i]; sxx += (double)xs[i]*xs[i]; sxy += (double)xs[i]*(double)ys[i]; }
    double mean_x = sx/n, mean_y = sy/n;
    double b = (sxy - n*mean_x*mean_y) / (sxx - n*mean_x*mean_x);
    double a = mean_y - b*mean_x;
    // R^2
    double ss_tot = 0, ss_res = 0;
    for (int i = 0; i < n; i++) {
        double pred = a + b*xs[i];
        ss_tot += (ys[i]-mean_y)*(ys[i]-mean_y);
        ss_res += (ys[i]-pred)*(ys[i]-pred);
    }
    double r2 = (ss_tot > 0) ? (1.0 - ss_res/ss_tot) : 1.0;
    return {a, b, r2};
}

int main() {
    CK(cudaSetDevice(0));

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    int clk_mhz = 0; cudaDeviceGetAttribute(&clk_mhz, cudaDevAttrClockRate, dev); // KHz
    printf("Device: %s, clockRate=%d KHz (= %.3f GHz)\n",
        p.name, clk_mhz, clk_mhz / 1e6);

    unsigned long long* d_out;
    unsigned* d_dummy;
    CK(cudaMalloc(&d_out, 64));
    CK(cudaMalloc(&d_dummy, 64));
    cudaMemset(d_dummy, 0, 64);
    cudaMemset(d_out, 0, 64);

    const std::vector<int> Ns = {0, 1, 2, 4, 8, 16, 32};
    const int trials = 21;

    const char* names[] = {"membar.cta", "membar.gl ", "membar.sys"};

    printf("\n=== V54 membar isolation: cy per N issued, single thread, median of %d trials ===\n", trials);
    printf("%-12s", "scope");
    for (int n : Ns) printf("%8d", n);
    printf("   |  per_fence (slope)   const (intercept)   R^2\n");
    printf("%-12s", "");
    for (size_t i = 0; i < Ns.size(); i++) printf("    N=%d", Ns[i]);
    printf("\n");

    // Three scopes
    for (int sc = 0; sc < 3; sc++) {
        std::vector<unsigned long long> ys(Ns.size());
        for (size_t i = 0; i < Ns.size(); i++) {
            unsigned long long cy;
            if      (sc == 0) cy = run_one<0>(Ns[i], d_out, d_dummy, trials);
            else if (sc == 1) cy = run_one<1>(Ns[i], d_out, d_dummy, trials);
            else              cy = run_one<2>(Ns[i], d_out, d_dummy, trials);
            ys[i] = cy;
        }

        // Print N -> cy
        printf("%-12s", names[sc]);
        for (auto y : ys) printf("%8llu", y);

        // Fit on N >= 1 (skip the N=0 outlier — it's pure overhead, no fences)
        std::vector<int> xs_fit(Ns.begin()+1, Ns.end());
        std::vector<unsigned long long> ys_fit(ys.begin()+1, ys.end());
        Fit f = linfit(xs_fit, ys_fit);

        // Also cy/issue at N=32 as a sanity check (should match slope if linear)
        double per_at_32 = (Ns.back() > 0) ? (double)(ys.back() - ys[0]) / Ns.back() : 0.0;

        printf("   |  %.2f cy/fence       %.2f               %.4f   (cy/issue@N=32 minus N=0: %.2f)\n",
            f.b, f.a, f.r2, per_at_32);
    }

    // ns conversion at boost
    double clk_ghz = clk_mhz / 1e6;
    printf("\n(For ns at this clock %.3f GHz, divide cy by %.3f.)\n", clk_ghz, clk_ghz);

    cudaFree(d_out); cudaFree(d_dummy);
    return 0;
}
