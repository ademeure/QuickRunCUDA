// dfma_pipelining.cu — Rigorous test of whether DFMA is zero-pipelined on B300
//
// The catalog claims DFMA has "zero ILP benefit" based on ILP=1 vs ILP=8.
// But DFMA latency is ~64 cy, so to pipeline you'd need ILP >= 64!
// This test sweeps ILP from 1 to 64 to find the true answer.
//
// Test structure:
//   ILP independent dep-chains of DFMA, each of length ITERS
//   Measure total cycles with clock64() at warp-granularity (thread 0)
//   cy/op = total_cycles / (ILP * ITERS)
//
// If zero-pipelined:  cy/op stays constant ~64 regardless of ILP
// If pipelined:       cy/op drops as ILP increases, flooring at throughput-limit
//
// Compile:  nvcc -arch=sm_103a -O3 -DILP=<N> -DITERS=1000 dfma_pipelining.cu -o dfma_test
// Lock:     nvidia-smi -lgc 2032
//
// ILP values tested: 1, 2, 4, 8, 16, 32, 64
// ITERS=1000 gives enough ops to average out overhead.
// We use asm volatile to prevent PTX optimizations.

#include <cstdint>
#include <cstdio>
#include <cmath>

#ifndef ILP
#define ILP 8
#endif

#ifndef ITERS
#define ITERS 1000
#endif

// Safety check - ILP must be 1..64
#if ILP < 1 || ILP > 64
#error "ILP must be 1..64"
#endif

// Helper to expand N independent chains
// We use macros to emit exactly ILP chains regardless of loop-unroll behavior.
// Each chain: a[k] = a[k] * b + c  (RAW dep on a[k])

// Returns packed result: upper 32 bits = cycle count top, lower 32 = cycle count low
// Also writes accumulator sum to *acc_out (anti-DCE)
__device__ __noinline__ unsigned long long dfma_chain(int iters, double b, double c, unsigned seed, double* acc_out) {
    // Initialize accumulators from seed so compiler can't prove values at compile time
    double a0  = 1.0 + 0.0001 * (double)(seed & 0xFF);
#if ILP >= 2
    double a1  = 1.5 + 0.0001 * (double)((seed>>1) & 0xFF);
#endif
#if ILP >= 4
    double a2  = 2.0 + 0.0001 * (double)((seed>>2) & 0xFF);
    double a3  = 2.5 + 0.0001 * (double)((seed>>3) & 0xFF);
#endif
#if ILP >= 8
    double a4  = 3.0 + 0.0001 * (double)((seed>>4) & 0xFF);
    double a5  = 3.5 + 0.0001 * (double)((seed>>5) & 0xFF);
    double a6  = 4.0 + 0.0001 * (double)((seed>>6) & 0xFF);
    double a7  = 4.5 + 0.0001 * (double)((seed>>7) & 0xFF);
#endif
#if ILP >= 16
    double a8  = 5.0 + 0.0001 * (double)((seed>>8)  & 0xFF);
    double a9  = 5.5 + 0.0001 * (double)((seed>>9)  & 0xFF);
    double a10 = 6.0 + 0.0001 * (double)((seed>>10) & 0xFF);
    double a11 = 6.5 + 0.0001 * (double)((seed>>11) & 0xFF);
    double a12 = 7.0 + 0.0001 * (double)((seed>>12) & 0xFF);
    double a13 = 7.5 + 0.0001 * (double)((seed>>13) & 0xFF);
    double a14 = 8.0 + 0.0001 * (double)((seed>>14) & 0xFF);
    double a15 = 8.5 + 0.0001 * (double)((seed>>15) & 0xFF);
#endif
#if ILP >= 32
    double a16 = 9.0  + 0.0001 * (double)((seed>>16) & 0xFF);
    double a17 = 9.5  + 0.0001 * (double)((seed>>17) & 0xFF);
    double a18 = 10.0 + 0.0001 * (double)((seed>>18) & 0xFF);
    double a19 = 10.5 + 0.0001 * (double)((seed>>19) & 0xFF);
    double a20 = 11.0 + 0.0001 * (double)((seed>>20) & 0xFF);
    double a21 = 11.5 + 0.0001 * (double)((seed>>21) & 0xFF);
    double a22 = 12.0 + 0.0001 * (double)((seed>>22) & 0xFF);
    double a23 = 12.5 + 0.0001 * (double)((seed>>23) & 0xFF);
    double a24 = 13.0 + 0.0001 * (double)((seed>>24) & 0xFF);
    double a25 = 13.5 + 0.0001 * (double)((seed>>25) & 0xFF);
    double a26 = 14.0 + 0.0001 * (double)((seed>>26) & 0xFF);
    double a27 = 14.5 + 0.0001 * (double)((seed>>27) & 0xFF);
    double a28 = 15.0 + 0.0001 * (double)((seed>>28) & 0xFF);
    double a29 = 15.5 + 0.0001 * (double)((seed>>29) & 0xFF);
    double a30 = 16.0 + 0.0001 * (double)((seed>>30) & 0xFF);
    double a31 = 16.5 + 0.0001 * (double)((seed>>31) & 0xFF);
#endif
#if ILP >= 64
    double a32 = 17.0 + 0.0001 * (double)((seed ^ (seed>>3)) & 0xFF);
    double a33 = 17.5 + 0.0001 * (double)((seed ^ (seed>>4)) & 0xFF);
    double a34 = 18.0 + 0.0001 * (double)((seed ^ (seed>>5)) & 0xFF);
    double a35 = 18.5 + 0.0001 * (double)((seed ^ (seed>>6)) & 0xFF);
    double a36 = 19.0 + 0.0001 * (double)((seed ^ (seed>>7)) & 0xFF);
    double a37 = 19.5 + 0.0001 * (double)((seed ^ (seed>>8)) & 0xFF);
    double a38 = 20.0 + 0.0001 * (double)((seed ^ (seed>>9)) & 0xFF);
    double a39 = 20.5 + 0.0001 * (double)((seed ^ (seed>>10)) & 0xFF);
    double a40 = 21.0 + 0.0001 * (double)((seed ^ (seed>>11)) & 0xFF);
    double a41 = 21.5 + 0.0001 * (double)((seed ^ (seed>>12)) & 0xFF);
    double a42 = 22.0 + 0.0001 * (double)((seed ^ (seed>>13)) & 0xFF);
    double a43 = 22.5 + 0.0001 * (double)((seed ^ (seed>>14)) & 0xFF);
    double a44 = 23.0 + 0.0001 * (double)((seed ^ (seed>>15)) & 0xFF);
    double a45 = 23.5 + 0.0001 * (double)((seed ^ (seed>>16)) & 0xFF);
    double a46 = 24.0 + 0.0001 * (double)((seed ^ (seed>>17)) & 0xFF);
    double a47 = 24.5 + 0.0001 * (double)((seed ^ (seed>>18)) & 0xFF);
    double a48 = 25.0 + 0.0001 * (double)((seed ^ (seed>>19)) & 0xFF);
    double a49 = 25.5 + 0.0001 * (double)((seed ^ (seed>>20)) & 0xFF);
    double a50 = 26.0 + 0.0001 * (double)((seed ^ (seed>>21)) & 0xFF);
    double a51 = 26.5 + 0.0001 * (double)((seed ^ (seed>>22)) & 0xFF);
    double a52 = 27.0 + 0.0001 * (double)((seed ^ (seed>>23)) & 0xFF);
    double a53 = 27.5 + 0.0001 * (double)((seed ^ (seed>>24)) & 0xFF);
    double a54 = 28.0 + 0.0001 * (double)((seed ^ (seed>>25)) & 0xFF);
    double a55 = 28.5 + 0.0001 * (double)((seed ^ (seed>>26)) & 0xFF);
    double a56 = 29.0 + 0.0001 * (double)((seed ^ (seed>>27)) & 0xFF);
    double a57 = 29.5 + 0.0001 * (double)((seed ^ (seed>>28)) & 0xFF);
    double a58 = 30.0 + 0.0001 * (double)((seed ^ (seed>>29)) & 0xFF);
    double a59 = 30.5 + 0.0001 * (double)((seed ^ (seed>>30)) & 0xFF);
    double a60 = 31.0 + 0.0001 * (double)((seed ^ (seed>>31)) & 0xFF);
    double a61 = 31.5 + 0.0001 * (double)((seed*3) & 0xFF) * 0.0001;
    double a62 = 32.0 + 0.0001 * (double)((seed*5) & 0xFF) * 0.0001;
    double a63 = 32.5 + 0.0001 * (double)((seed*7) & 0xFF) * 0.0001;
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

    // Anti-DCE: accumulate and conditionally store
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
    *acc_out = acc;  // Write to caller-provided pointer — prevents DCE
    return t1 - t0;
}

extern "C" __global__ void kernel(float* A, float* B, float* C,
                                   int iters, int arg1, int arg2)
{
    // Only thread 0 of each block measures — avoids warp divergence in timing
    // but still fills SM with a representative warp.
    // We run with 32 threads (1 warp) per block so timing is unambiguous.

    unsigned seed = blockIdx.x * 31 + threadIdx.x + 1;
    double b = 1.0 + 1e-6 * (double)((seed) & 0xFF);
    double c = 1e-7 * (double)((seed*3) & 0xFF);

    double acc_result = 0.0;
    unsigned long long dt = dfma_chain(iters, b, c, seed, &acc_result);

    // thread 0 of each block stores the cycle count
    if (threadIdx.x == 0) {
        // Store raw cycles into C (as bits)
        unsigned long long* C64 = (unsigned long long*)C;
        C64[blockIdx.x] = dt;
        // Anti-DCE: store accumulator sum to B if impossible condition (compiler can't fold)
        if (acc_result == (double)0xDEADBEEFu) {
            ((double*)B)[blockIdx.x] = acc_result;
        }
    }
}
