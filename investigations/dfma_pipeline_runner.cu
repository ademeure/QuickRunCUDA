// dfma_pipeline_runner.cu — standalone host+device program
// Tests DFMA pipelining at ILP=1,2,4,8,16,32,64
// Compile: nvcc -arch=sm_103a -O3 -DILP=<N> -DITERS=2000 dfma_pipeline_runner.cu -o dfma_ilp<N>
// Run:     ./dfma_ilp<N>
// Lock:    nvidia-smi -lgc 2032

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <climits>
#include <algorithm>

#ifndef ILP
#define ILP 8
#endif

#ifndef ITERS
#define ITERS 2000
#endif

// How many blocks to run:
// Single-SM test: 1 block, 32 threads
// Full-GPU test:  148 blocks, 32 threads each
#ifndef NBLOCKS
#define NBLOCKS 1
#endif

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Device kernel: ILP independent DFMA chains, timed with clock64
// ============================================================

__device__ __noinline__ unsigned long long dfma_chain_timed(
    int iters, double b, double c, unsigned seed, double* acc_out)
{
    // Initialize each chain accumulator with a runtime-varying value.
    // Using (seed + k*prime) to ensure different registers are allocated.
    double a0  = 1.0 + 0.0001 * (double)(seed & 0xFFF);
#if ILP >= 2
    double a1  = 1.5 + 0.0001 * (double)((seed+7) & 0xFFF);
#endif
#if ILP >= 4
    double a2  = 2.0 + 0.0001 * (double)((seed+13) & 0xFFF);
    double a3  = 2.5 + 0.0001 * (double)((seed+19) & 0xFFF);
#endif
#if ILP >= 8
    double a4  = 3.0 + 0.0001 * (double)((seed+23) & 0xFFF);
    double a5  = 3.5 + 0.0001 * (double)((seed+29) & 0xFFF);
    double a6  = 4.0 + 0.0001 * (double)((seed+31) & 0xFFF);
    double a7  = 4.5 + 0.0001 * (double)((seed+37) & 0xFFF);
#endif
#if ILP >= 16
    double a8  = 5.0 + 0.0001 * (double)((seed+41) & 0xFFF);
    double a9  = 5.5 + 0.0001 * (double)((seed+43) & 0xFFF);
    double a10 = 6.0 + 0.0001 * (double)((seed+47) & 0xFFF);
    double a11 = 6.5 + 0.0001 * (double)((seed+53) & 0xFFF);
    double a12 = 7.0 + 0.0001 * (double)((seed+59) & 0xFFF);
    double a13 = 7.5 + 0.0001 * (double)((seed+61) & 0xFFF);
    double a14 = 8.0 + 0.0001 * (double)((seed+67) & 0xFFF);
    double a15 = 8.5 + 0.0001 * (double)((seed+71) & 0xFFF);
#endif
#if ILP >= 32
    double a16 = 9.0  + 0.0001 * (double)((seed+73)  & 0xFFF);
    double a17 = 9.5  + 0.0001 * (double)((seed+79)  & 0xFFF);
    double a18 = 10.0 + 0.0001 * (double)((seed+83)  & 0xFFF);
    double a19 = 10.5 + 0.0001 * (double)((seed+89)  & 0xFFF);
    double a20 = 11.0 + 0.0001 * (double)((seed+97)  & 0xFFF);
    double a21 = 11.5 + 0.0001 * (double)((seed+101) & 0xFFF);
    double a22 = 12.0 + 0.0001 * (double)((seed+103) & 0xFFF);
    double a23 = 12.5 + 0.0001 * (double)((seed+107) & 0xFFF);
    double a24 = 13.0 + 0.0001 * (double)((seed+109) & 0xFFF);
    double a25 = 13.5 + 0.0001 * (double)((seed+113) & 0xFFF);
    double a26 = 14.0 + 0.0001 * (double)((seed+127) & 0xFFF);
    double a27 = 14.5 + 0.0001 * (double)((seed+131) & 0xFFF);
    double a28 = 15.0 + 0.0001 * (double)((seed+137) & 0xFFF);
    double a29 = 15.5 + 0.0001 * (double)((seed+139) & 0xFFF);
    double a30 = 16.0 + 0.0001 * (double)((seed+149) & 0xFFF);
    double a31 = 16.5 + 0.0001 * (double)((seed+151) & 0xFFF);
#endif
#if ILP >= 64
    double a32 = 17.0 + 0.0001 * (double)((seed+157) & 0xFFF);
    double a33 = 17.5 + 0.0001 * (double)((seed+163) & 0xFFF);
    double a34 = 18.0 + 0.0001 * (double)((seed+167) & 0xFFF);
    double a35 = 18.5 + 0.0001 * (double)((seed+173) & 0xFFF);
    double a36 = 19.0 + 0.0001 * (double)((seed+179) & 0xFFF);
    double a37 = 19.5 + 0.0001 * (double)((seed+181) & 0xFFF);
    double a38 = 20.0 + 0.0001 * (double)((seed+191) & 0xFFF);
    double a39 = 20.5 + 0.0001 * (double)((seed+193) & 0xFFF);
    double a40 = 21.0 + 0.0001 * (double)((seed+197) & 0xFFF);
    double a41 = 21.5 + 0.0001 * (double)((seed+199) & 0xFFF);
    double a42 = 22.0 + 0.0001 * (double)((seed+211) & 0xFFF);
    double a43 = 22.5 + 0.0001 * (double)((seed+223) & 0xFFF);
    double a44 = 23.0 + 0.0001 * (double)((seed+227) & 0xFFF);
    double a45 = 23.5 + 0.0001 * (double)((seed+229) & 0xFFF);
    double a46 = 24.0 + 0.0001 * (double)((seed+233) & 0xFFF);
    double a47 = 24.5 + 0.0001 * (double)((seed+239) & 0xFFF);
    double a48 = 25.0 + 0.0001 * (double)((seed+241) & 0xFFF);
    double a49 = 25.5 + 0.0001 * (double)((seed+251) & 0xFFF);
    double a50 = 26.0 + 0.0001 * (double)((seed+257) & 0xFFF);
    double a51 = 26.5 + 0.0001 * (double)((seed+263) & 0xFFF);
    double a52 = 27.0 + 0.0001 * (double)((seed+269) & 0xFFF);
    double a53 = 27.5 + 0.0001 * (double)((seed+271) & 0xFFF);
    double a54 = 28.0 + 0.0001 * (double)((seed+277) & 0xFFF);
    double a55 = 28.5 + 0.0001 * (double)((seed+281) & 0xFFF);
    double a56 = 29.0 + 0.0001 * (double)((seed+283) & 0xFFF);
    double a57 = 29.5 + 0.0001 * (double)((seed+293) & 0xFFF);
    double a58 = 30.0 + 0.0001 * (double)((seed+307) & 0xFFF);
    double a59 = 30.5 + 0.0001 * (double)((seed+311) & 0xFFF);
    double a60 = 31.0 + 0.0001 * (double)((seed+313) & 0xFFF);
    double a61 = 31.5 + 0.0001 * (double)((seed+317) & 0xFFF);
    double a62 = 32.0 + 0.0001 * (double)((seed+331) & 0xFFF);
    double a63 = 32.5 + 0.0001 * (double)((seed+337) & 0xFFF);
#endif

    unsigned long long t0 = clock64();

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a0) : "d"(b), "d"(c));
#if ILP >= 2
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a1) : "d"(b), "d"(c));
#endif
#if ILP >= 4
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a2) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a3) : "d"(b), "d"(c));
#endif
#if ILP >= 8
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a4) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a5) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a6) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a7) : "d"(b), "d"(c));
#endif
#if ILP >= 16
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a8)  : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a9)  : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a10) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a11) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a12) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a13) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a14) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a15) : "d"(b), "d"(c));
#endif
#if ILP >= 32
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a16) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a17) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a18) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a19) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a20) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a21) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a22) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a23) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a24) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a25) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a26) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a27) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a28) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a29) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a30) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a31) : "d"(b), "d"(c));
#endif
#if ILP >= 64
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a32) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a33) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a34) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a35) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a36) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a37) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a38) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a39) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a40) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a41) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a42) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a43) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a44) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a45) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a46) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a47) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a48) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a49) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a50) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a51) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a52) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a53) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a54) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a55) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a56) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a57) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a58) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a59) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a60) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a61) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a62) : "d"(b), "d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a63) : "d"(b), "d"(c));
#endif
    }

    unsigned long long t1 = clock64();

    // Accumulate all chains — anti-DCE
    double acc = a0;
#if ILP >= 2
    acc += a1;
#endif
#if ILP >= 4
    acc += a2 + a3;
#endif
#if ILP >= 8
    acc += a4 + a5 + a6 + a7;
#endif
#if ILP >= 16
    acc += a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
#endif
#if ILP >= 32
    acc += a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23
         + a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31;
#endif
#if ILP >= 64
    acc += a32 + a33 + a34 + a35 + a36 + a37 + a38 + a39
         + a40 + a41 + a42 + a43 + a44 + a45 + a46 + a47
         + a48 + a49 + a50 + a51 + a52 + a53 + a54 + a55
         + a56 + a57 + a58 + a59 + a60 + a61 + a62 + a63;
#endif
    *acc_out = acc;
    return t1 - t0;
}

__global__ void bench_kernel(unsigned long long* cycle_out, double* dummy_out, int nblocks)
{
    // Each block = 1 warp (32 threads). Only thread 0 does timing.
    unsigned seed = blockIdx.x * 31337u + threadIdx.x + 1;
    double b = 1.0 + 1e-7 * (double)(seed & 0xFFF);
    double c = 1e-8 * (double)((seed * 1234567) & 0xFFF);

    double acc = 0.0;
    unsigned long long dt = dfma_chain_timed(ITERS, b, c, seed, &acc);

    if (threadIdx.x == 0) {
        cycle_out[blockIdx.x] = dt;
    }
    // Anti-DCE for all threads: write acc to dummy if impossible
    if (acc == (double)0xDEADBEEFULL) {
        dummy_out[blockIdx.x * 32 + threadIdx.x] = acc;
    }
}

int main(int argc, char** argv)
{
    int nblocks = NBLOCKS;
    if (argc > 1) nblocks = atoi(argv[1]);

    // Query SM count
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    printf("DFMA Pipelining Test: ILP=%d, ITERS=%d, blocks=%d (SM_count=%d)\n",
           ILP, ITERS, nblocks, sm_count);

    unsigned long long* d_cycles;
    double* d_dummy;
    CUDA_CHECK(cudaMalloc(&d_cycles, nblocks * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_dummy, nblocks * 32 * sizeof(double)));

    // Warmup
    bench_kernel<<<nblocks, 32>>>(d_cycles, d_dummy, nblocks);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs — take minimum across 5 launches (for clock64, minimum = best warp)
    unsigned long long* h_cycles = new unsigned long long[nblocks];
    unsigned long long best_min = ULLONG_MAX;
    unsigned long long best_max = 0;
    double sum_min = 0;

    for (int run = 0; run < 5; run++) {
        bench_kernel<<<nblocks, 32>>>(d_cycles, d_dummy, nblocks);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_cycles, d_cycles, nblocks * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));

        unsigned long long rmin = ULLONG_MAX, rmax = 0;
        double rsum = 0;
        for (int b = 0; b < nblocks; b++) {
            rmin = std::min(rmin, h_cycles[b]);
            rmax = std::max(rmax, h_cycles[b]);
            rsum += (double)h_cycles[b];
        }
        if (run == 0 || rmin < best_min) {
            best_min = rmin;
            best_max = rmax;
            sum_min = rsum;
        }
        printf("  run %d: min_cy=%llu  max_cy=%llu  mean_cy=%.1f\n",
               run, rmin, rmax, rsum / nblocks);
    }

    // Compute cycles per DFMA
    // total DFMAs = ILP * ITERS (per warp/block)
    long long total_dfmas = (long long)ILP * ITERS;
    double cy_per_dfma_min  = (double)best_min  / total_dfmas;
    double cy_per_dfma_max  = (double)best_max  / total_dfmas;
    double cy_per_dfma_mean = sum_min / nblocks  / total_dfmas;

    printf("\n=== ILP=%d ITERS=%d blocks=%d ===\n", ILP, ITERS, nblocks);
    printf("  total_dfma_per_warp = %lld\n", total_dfmas);
    printf("  cy/DFMA (best block, min across runs): %.3f\n", cy_per_dfma_min);
    printf("  cy/DFMA (worst block, same run):       %.3f\n", cy_per_dfma_max);
    printf("  cy/DFMA (mean across blocks):          %.3f\n", cy_per_dfma_mean);

    // Theoretical: if fully pipelined, cy/DFMA_throughput = latency / ILP (floor at 1/dispatch_rate)
    // If zero-pipelined: cy/DFMA = latency ~= 64 cy
    printf("\n  Prediction if ZERO-pipelined: cy/DFMA ~ 64 (constant regardless of ILP)\n");
    printf("  Prediction if PIPELINED:       cy/DFMA = max(64/ILP, throughput_floor)\n");
    printf("    ILP=%d: expected %.1f cy/DFMA if pipelined (throughput = %.1f)\n",
           ILP, 64.0 / ILP, 64.0 / ILP);

    delete[] h_cycles;
    cudaFree(d_cycles);
    cudaFree(d_dummy);
    return 0;
}
