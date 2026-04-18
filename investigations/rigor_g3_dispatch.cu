// G3 RIGOR: block dispatch tail latency formula
//
// THEORETICAL: each block must be dispatched to an SM by the scheduler.
// At full occupancy, 8 blocks/SM × 148 SMs = 1184 in-flight. New blocks
// dispatch as old ones finish. If block runtime < dispatch_latency,
// warp slots go empty between blocks → warps_active < sm_active.
//
// Test: vary per-block work (FFMA iters), measure achieved TFLOPS.
// At low iters, throughput drops below peak even though SMs busy.

#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void short_block(float *out, float a, int iters) {
    float r0 = 0.5f, r1 = 1.5f, r2 = 2.5f, r3 = 3.5f;
    float r4 = 4.5f, r5 = 5.5f, r6 = 6.5f, r7 = 7.5f;
    float b = a + 1.0f, c = a + 2.0f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        r0 = r0 * a + b;  r1 = r1 * a + c;  r2 = r2 * a + b;  r3 = r3 * a + c;
        r4 = r4 * a + b;  r5 = r5 * a + c;  r6 = r6 * a + b;  r7 = r7 * a + c;
    }
    float sum = r0+r1+r2+r3+r4+r5+r6+r7;
    if (sum < -1e30f) out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 1 << 24);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int blocks = 148 * 8 * 16;  // 16 waves
    int threads = 256;

    printf("# Block FFMA iters → achieved TFLOPS (148 × 8 × 16 = %d blocks)\n", blocks);
    printf("# iters     time(ms)  TFLOPS  per-block-ns\n");
    int iters_list[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    for (int iters : iters_list) {
        for (int i = 0; i < 3; i++) short_block<<<blocks, threads>>>(d_out, 1.5f, iters);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            short_block<<<blocks, threads>>>(d_out, 1.5f, iters);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long total_ffma = (long)blocks * threads * iters * 8;
        double tflops = total_ffma * 2.0 / (best/1000) / 1e12;
        // per-block runtime ≈ best / 16 waves
        double per_block_ns = best * 1e6 / 16;
        printf("  %5d   %6.3f   %5.2f   %.0f\n",
               iters, best, tflops, per_block_ns);
    }
    return 0;
}
