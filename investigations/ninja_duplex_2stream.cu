// NINJA: 2-stream concurrent R-only + W-only kernels — test HBM3E true duplex
// If HBM3E is fully duplex per stack: aggregate ~ 7.31 (R) + 7.57 (W) = 14.88 TB/s
// If partial duplex: somewhere between 6.68 (single mixed) and 14.88
// Already saw copy = 6.93 → suggests channels CAN partial-duplex when separated

#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void w_only(int *data, int v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *p = data + (warp_id * 32 + lane) * 8;
    asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
        :: "l"(p), "r"(v) : "memory");
}

__launch_bounds__(256, 8) __global__ void r_only(const int *data, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    const int *base = data + warp_id * (2 * 32 * 8);  // 2 KB per warp = best read recipe
    int acc = 0;
    #pragma unroll
    for (int it = 0; it < 2; it++) {
        const int *p = base + (it * 32 + lane) * 8;
        int r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "l"(p));
        acc ^= r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7;
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
    else if (tid == 0) out[0] = acc;
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 4ull * 1024 * 1024 * 1024;
    int *d_r, *d_w, *d_out;
    cudaMalloc(&d_r, bytes); cudaMemset(d_r, 0xab, bytes);
    cudaMalloc(&d_w, bytes);
    cudaMalloc(&d_out, 1<<24);

    cudaStream_t sR, sW, sBoth;
    cudaStreamCreateWithFlags(&sR, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&sW, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&sBoth, cudaStreamNonBlocking);

    cudaEvent_t e0, e1, eR0, eR1, eW0, eW1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&eR0); cudaEventCreate(&eR1);
    cudaEventCreate(&eW0); cudaEventCreate(&eW1);

    int blocks_w = bytes / (256 * 32);
    int blocks_r = bytes / (256 * 64);

    // Baseline: each kernel alone
    auto bench_alone = [&](auto launch, const char *label, size_t b) {
        for (int i = 0; i < 5; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 30; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double gbs = b / (best/1000) / 1e9;
        printf("  %s alone: %.4f ms = %.0f GB/s = %.2f%% spec\n",
               label, best, gbs, gbs/7672*100);
    };

    bench_alone([&]{ w_only<<<blocks_w, 256>>>(d_w, 0xab); }, "WRITE", bytes);
    bench_alone([&]{ r_only<<<blocks_r, 256>>>(d_r, d_out); }, "READ ", bytes);

    // Concurrent: R on stream A, W on stream B. Time the LATER-ending kernel.
    printf("\n# Concurrent R + W (2 streams):\n");
    for (int j = 0; j < 5; j++) {  // 5 outer reps
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        cudaEventRecord(eR0, sR);
        r_only<<<blocks_r, 256, 0, sR>>>(d_r, d_out);
        cudaEventRecord(eR1, sR);
        cudaEventRecord(eW0, sW);
        w_only<<<blocks_w, 256, 0, sW>>>(d_w, 0xcd);
        cudaEventRecord(eW1, sW);
        cudaEventSynchronize(eR1);
        cudaEventSynchronize(eW1);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms_r, ms_w, ms_total;
        cudaEventElapsedTime(&ms_r, eR0, eR1);
        cudaEventElapsedTime(&ms_w, eW0, eW1);
        cudaEventElapsedTime(&ms_total, e0, e1);
        double r_gbs = bytes / (ms_r/1000) / 1e9;
        double w_gbs = bytes / (ms_w/1000) / 1e9;
        double agg_total = 2.0 * bytes / (ms_total/1000) / 1e9;
        printf("  rep %d: R=%.0f, W=%.0f GB/s; total wall %.4f ms; AGGREGATE=%.0f GB/s = %.2f%% spec\n",
               j, r_gbs, w_gbs, ms_total, agg_total, agg_total/7672*100);
    }
    return 0;
}
