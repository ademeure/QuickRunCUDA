// High-ILP DFMA test: ILP=128, 256 to see if throughput ever improves beyond 64 cy/op
// Uses 128 or 256 independent chains
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <climits>
#include <algorithm>

#ifndef ILP
#define ILP 128
#endif
#ifndef ITERS
#define ITERS 500
#endif
#ifndef NBLOCKS
#define NBLOCKS 1
#endif

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %d\n",cudaGetErrorString(e),__LINE__);exit(1);}} while(0)

// Use PTX loop to avoid having to unroll N chains at C level
// This uses a trick: store N chains in shared memory, access via ld.shared
// to prevent the compiler from treating them as registers

__global__ void bench_kernel(unsigned long long* cycle_out, double* dummy, int nb)
{
    // We use N shared-memory double accumulators
    // Each thread has its own set (no inter-thread sharing)
    // blockDim.x = 32 (1 warp)
    __shared__ double accs[32 * ILP];   // 32 threads × ILP chains

    unsigned seed = blockIdx.x * 31337u + threadIdx.x + 1;
    double b = 1.0 + 1e-7 * (double)(seed & 0xFFF);
    double c = 1e-8 * (double)((seed*1234567) & 0xFFF);

    // Initialize all chains
    int base = threadIdx.x * ILP;
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        accs[base + k] = 1.0 + 0.0001 * (double)((seed + k*7) & 0xFFF);
    }
    __syncthreads();

    unsigned long long t0 = clock64();

    // Main loop: ILP sequential DFMAs per iteration
    // Using shared memory to force register pressure relief
    // But this adds memory latency... 
    // Alternative: use PTX directly with many register variables
    // For ILP>64 we need a different approach
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            double a = accs[base + k];
            asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a) : "d"(b), "d"(c));
            accs[base + k] = a;
        }
    }
    __syncthreads();

    unsigned long long t1 = clock64();

    if (threadIdx.x == 0) {
        cycle_out[blockIdx.x] = t1 - t0;
    }
    double acc_sum = 0;
    for (int k = 0; k < ILP; k++) acc_sum += accs[base + k];
    if (acc_sum == (double)0xDEADBEEFULL) dummy[blockIdx.x*32+threadIdx.x] = acc_sum;
}

int main(int argc, char** argv) {
    int nblocks = NBLOCKS;
    if (argc>1) nblocks = atoi(argv[1]);

    printf("HIGH-ILP DFMA: ILP=%d ITERS=%d blocks=%d\n", ILP, ITERS, nblocks);

    unsigned long long* d_cyc; double* d_dum;
    CUDA_CHECK(cudaMalloc(&d_cyc, nblocks*sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_dum, nblocks*32*sizeof(double)));

    bench_kernel<<<nblocks,32>>>(d_cyc, d_dum, nblocks);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long* h_cyc = new unsigned long long[nblocks];
    unsigned long long best_min = ULLONG_MAX;
    for (int r=0; r<5; r++) {
        bench_kernel<<<nblocks,32>>>(d_cyc, d_dum, nblocks);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_cyc, d_cyc, nblocks*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        unsigned long long rmin = h_cyc[0];
        for(int b=0;b<nblocks;b++) rmin = std::min(rmin, h_cyc[b]);
        best_min = std::min(best_min, rmin);
        printf("  run %d: min_cy=%llu\n", r, rmin);
    }
    long long total = (long long)ILP * ITERS;
    printf("  cy/DFMA: %.3f\n", (double)best_min / total);
    printf("  (Note: shared-memory version adds smem latency overhead)\n");
    delete[] h_cyc;
    return 0;
}
