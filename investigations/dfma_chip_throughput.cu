// dfma_chip_throughput.cu — measures DFMA total chip throughput
// Uses CUDA events for wall-clock measurement (not clock64 per warp)
// Sweeps warps/SM to find peak FP64 TFLOPS

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA error %s\n",cudaGetErrorString(e));exit(1);}} while(0)

#ifndef ITERS
#define ITERS 10000
#endif

__global__ void dfma_kernel(double* dummy, int iters) {
    unsigned seed = blockIdx.x * 31337u + threadIdx.x + 1;
    double a0 = 1.0 + 0.0001*(seed & 0xFF);
    double a1 = 1.5 + 0.0001*((seed>>1) & 0xFF);
    double a2 = 2.0 + 0.0001*((seed>>2) & 0xFF);
    double a3 = 2.5 + 0.0001*((seed>>3) & 0xFF);
    double a4 = 3.0 + 0.0001*((seed>>4) & 0xFF);
    double a5 = 3.5 + 0.0001*((seed>>5) & 0xFF);
    double a6 = 4.0 + 0.0001*((seed>>6) & 0xFF);
    double a7 = 4.5 + 0.0001*((seed>>7) & 0xFF);
    double b = 1.0 + 1e-7*(seed & 0xFFF);
    double c = 1e-8*((seed*1234567) & 0xFFF);

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        asm volatile("fma.rn.f64 %0,%0,%1,%2;":"+d"(a0):"d"(b),"d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;":"+d"(a1):"d"(b),"d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;":"+d"(a2):"d"(b),"d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;":"+d"(a3):"d"(b),"d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;":"+d"(a4):"d"(b),"d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;":"+d"(a5):"d"(b),"d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;":"+d"(a6):"d"(b),"d"(c));
        asm volatile("fma.rn.f64 %0,%0,%1,%2;":"+d"(a7):"d"(b),"d"(c));
    }
    if (a0+a1+a2+a3+a4+a5+a6+a7 == 0.0) dummy[blockIdx.x*blockDim.x+threadIdx.x] = a0;
}

int main() {
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    printf("FP64 DFMA chip throughput sweep (SM_count=%d, ILP=8, ITERS=%d)\n", sm_count, ITERS);
    printf("%-15s %-10s %-12s %-12s %-15s\n", "Blocks", "Warps/SM", "ms", "GFLOPS", "TFLOPS");

    double* d_dummy;
    CUDA_CHECK(cudaMalloc(&d_dummy, sm_count * 64 * 32 * sizeof(double)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // DFMAs per thread = 8 chains × ITERS × 2 FLOPS each
    long long flops_per_thread = 8LL * ITERS * 2;

    for (int warps_per_sm = 1; warps_per_sm <= 64; warps_per_sm *= 2) {
        int nblocks = sm_count * warps_per_sm;
        int nthreads = 32;  // 1 warp per block

        // Warmup
        dfma_kernel<<<nblocks, nthreads>>>(d_dummy, ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed
        float best_ms = 1e10f;
        for (int r = 0; r < 3; r++) {
            cudaEventRecord(t0);
            dfma_kernel<<<nblocks, nthreads>>>(d_dummy, ITERS);
            cudaEventRecord(t1);
            CUDA_CHECK(cudaDeviceSynchronize());
            float ms;
            cudaEventElapsedTime(&ms, t0, t1);
            if (ms < best_ms) best_ms = ms;
        }

        long long total_flops = flops_per_thread * (long long)nblocks * nthreads;
        double gflops = total_flops / (best_ms * 1e6);
        printf("%-15d %-10d %-12.3f %-12.1f %-15.3f\n",
               nblocks, warps_per_sm, best_ms, gflops, gflops/1000.0);
    }

    cudaFree(d_dummy);
    return 0;
}
