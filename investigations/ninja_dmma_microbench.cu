// Direct mma.sync.m16n8k4.f64 microbench - verify catalog's "~2 TFLOPS" claim
//
// Theoretical:
//   - mma.sync.m16n8k4.f64.f64.f64.f64: 1 warp computes m16*n8 = 128 outputs per MMA
//   - Per MMA: 16*8*4*2 = 1024 FLOPS for K=4 inner accumulate
//   - If pipe issues 1 mma per N cycles per warp: TFLOPS = 1024 * (clock/N) / 1e12
//   - For N=8 cy/MMA per warp, 4 SMSPs, 148 SMs, 2.032 GHz:
//     = 1024 * (2.032e9/8) * 4 * 148 / 1e12 = 154 TFLOPS (best case 8cy)
//   - For N=64 cy/MMA per warp = 19 TFLOPS
//   - Catalog "~2 TFLOPS" implies N=512 cy/MMA — extremely throttled
//
// DCE defense: ILP+keep, write final to global if impossible cond
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

template <int N_ITERS>
__launch_bounds__(128, 4) __global__ void dmma_loop(double *out, int N) {
    int warp_lane = threadIdx.x & 31;
    // m16n8k4 with f64: A=2/thr, B=1/thr, C=4/thr, D=4/thr
    double a0 = (double)warp_lane * 0.001;
    double a1 = (double)warp_lane * 0.002;
    double b0 = (double)warp_lane * 0.003;
    double c0 = (double)warp_lane * 0.004;
    double c1 = (double)warp_lane * 0.005;
    double c2 = (double)warp_lane * 0.006;
    double c3 = (double)warp_lane * 0.007;

    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_ITERS; j++) {
            asm volatile(
                "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                : "=d"(c0), "=d"(c1), "=d"(c2), "=d"(c3)
                : "d"(a0), "d"(a1), "d"(b0),
                  "d"(c0), "d"(c1), "d"(c2), "d"(c3)
            );
        }
    }
    if (c0 + c1 + c2 + c3 == 1e30) out[blockIdx.x * blockDim.x + threadIdx.x] = c0;
}

template <int N_ITERS>
double bench(int blocks, int N) {
    double *d_out;
    cudaMalloc(&d_out, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    // Warmup
    for (int i = 0; i < 3; i++) dmma_loop<N_ITERS><<<blocks, 128>>>(d_out, N);
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        dmma_loop<N_ITERS><<<blocks, 128>>>(d_out, N);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    // Total MMAs: blocks * 4 warps/block * N * N_ITERS
    long total_mmas = (long)blocks * 4 * N * N_ITERS;
    long total_flops = total_mmas * 16 * 8 * 4 * 2L;  // m16 n8 k4 fma=2
    double tflops = total_flops / (best/1000.0) / 1e12;
    cudaFree(d_out);
    return tflops;
}

int main() {
    cudaSetDevice(0);
    // 4 warps/block × 148 blocks/SM × 4 occ = ~592 blocks
    printf("# DMMA m16n8k4 f64 throughput sweep\n");
    int blocks = 148 * 4;  // 4 blocks per SM
    int N = 1000;
    printf("# blocks=%d N=%d N_ITERS varies\n", blocks, N);
    printf("ILP=1: %.2f TFLOPS\n", bench<1>(blocks, N));
    printf("ILP=2: %.2f TFLOPS\n", bench<2>(blocks, N));
    printf("ILP=4: %.2f TFLOPS\n", bench<4>(blocks, N));
    printf("ILP=8: %.2f TFLOPS\n", bench<8>(blocks, N));
    printf("ILP=16: %.2f TFLOPS\n", bench<16>(blocks, N));
    return 0;
}
