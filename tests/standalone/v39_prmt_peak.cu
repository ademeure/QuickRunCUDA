// V39: prmt (byte permute) peak — used heavily in FP4/FP8 conversions
// Theoretical: ALU pipe ~1 inst/cy/SMSP × 4 SMSPs × 148 SMs × 2.032 GHz = 1202 Ginst/s
// Per-byte: × 4 bytes/inst = 4808 Gbytes/s = 4.8 TB/s permute throughput

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int ILP, int N_ITERS>
__global__ __launch_bounds__(128, 2)
void prmt_peak(unsigned* out) {
    int tid = threadIdx.x;
    unsigned a[8], b[8], v[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        a[k] = (tid * 7 + k * 13) ^ (blockIdx.x * 37);
        b[k] = a[k] ^ 0xa5a5a5a5;
        v[k] = 0;
    }
    unsigned ctrl = 0x6420;  // some byte permutation pattern

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // True ILP: each chain depends on its own previous result
    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            asm volatile("prmt.b32 %0, %0, %1, %2;"
                         : "+r"(a[k]) : "r"(b[k]), "r"(ctrl));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (tid == 0) {
        unsigned acc = 0;
        for (int k = 0; k < ILP; k++) acc ^= a[k];
        out[blockIdx.x] = acc + (unsigned)(t1 - t0);
    }
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    unsigned* d_out;
    CK(cudaMalloc(&d_out, 4 * prop.multiProcessorCount * 2));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_ITERS = 5000;
    // Theoretical: 1 warp-inst/cy/SMSP × 32 lanes × 4 SMSPs × 148 SMs × 2.032 GHz
    const double THEO_GLANE = 148.0 * 4 * 32 * 2.032;

    printf("=== V39 prmt.b32 peak throughput (lane-level) ===\n");
    printf("ILP  blocks  wall_ms   total_lane   Glane/s  Gbytes/s   %%SoL\n");

    auto run = [&](int ilp, int blocks) {
        if (ilp == 1) prmt_peak<1, N_ITERS><<<blocks, 128>>>(d_out);
        else if (ilp == 4) prmt_peak<4, N_ITERS><<<blocks, 128>>>(d_out);
        else if (ilp == 8) prmt_peak<8, N_ITERS><<<blocks, 128>>>(d_out);
        cudaDeviceSynchronize();
        int RUNS = 3;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            if (ilp == 1) prmt_peak<1, N_ITERS><<<blocks, 128>>>(d_out);
            else if (ilp == 4) prmt_peak<4, N_ITERS><<<blocks, 128>>>(d_out);
            else if (ilp == 8) prmt_peak<8, N_ITERS><<<blocks, 128>>>(d_out);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        double total = (double)blocks * 128 * ilp * N_ITERS;  // 128 threads/CTA × ILP per thread
        double ginst = total / (avg_ms / 1e3) / 1e9;
        double gbytes = ginst * 4;
        printf("%d    %4d    %7.3f   %10.2fG   %7.2f   %7.2f    %.1f%%\n",
               ilp, blocks, avg_ms, total / 1e9, ginst, gbytes, ginst / THEO_GLANE * 100);
    };

    for (int b : {148, 296, 592, 1184}) run(1, b);
    for (int b : {148, 296, 592, 1184}) run(4, b);
    for (int b : {148, 296, 592, 1184}) run(8, b);

    printf("\nTheoretical (lane peak): %.2f Glanes/s = %.2f Gbytes/s\n",
           THEO_GLANE, THEO_GLANE * 4);

    cudaFree(d_out);
    return 0;
}
