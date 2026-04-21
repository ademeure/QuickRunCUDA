// V41: MUFU pipe — sin/cos/exp/log/rcp/rsqrt/sqrt comparison
// Theoretical: MUFU pipe = 1 inst per 4 cy per SMSP (V37/V38 pattern)
// Per-SM throughput: 4 SMSPs × (1/4) = 1 inst/cy per SM
// At 2032 MHz × 148 SMs × 32 lanes = 9.62 Gops/s lane-rate

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int OP, int ILP, int N_ITERS>
__global__ __launch_bounds__(128, 2)
void mufu_peak(float* out) {
    int tid = threadIdx.x;
    float v[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) v[k] = (tid + k * 0.13f) * 1.001f + (blockIdx.x * 0.001f);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            // %0 in-out: dep chain to prevent LICM, force ILP via separate slots
            if (OP == 0) asm volatile("ex2.approx.ftz.f32 %0, %0;"  : "+f"(v[k]));
            if (OP == 1) asm volatile("lg2.approx.ftz.f32 %0, %0;"  : "+f"(v[k]));
            if (OP == 2) asm volatile("rcp.approx.ftz.f32 %0, %0;"  : "+f"(v[k]));
            if (OP == 3) asm volatile("rsqrt.approx.ftz.f32 %0, %0;": "+f"(v[k]));
            if (OP == 4) asm volatile("sqrt.approx.ftz.f32 %0, %0;" : "+f"(v[k]));
            if (OP == 5) asm volatile("sin.approx.ftz.f32 %0, %0;"  : "+f"(v[k]));
            if (OP == 6) asm volatile("cos.approx.ftz.f32 %0, %0;"  : "+f"(v[k]));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (tid == 0) {
        float acc = 0;
        for (int k = 0; k < ILP; k++) acc += v[k];
        out[blockIdx.x] = acc + (float)(t1 - t0);
    }
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    float* d_out;
    CK(cudaMalloc(&d_out, 4 * prop.multiProcessorCount * 4));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_ITERS = 5000;
    const int BLOCKS = 1184;
    const int ILP = 8;
    // Theoretical: assume 1 inst per 4 cy per SMSP (V37 pattern)
    // = 148 × 4 SMSPs × (1/4) inst/cy × 32 lanes × 2.032 GHz
    const double THEO_GLANE = 148.0 * 4 * (1.0/4) * 32 * 2.032;

    printf("=== V41 MUFU pipe peak comparison (BLOCKS=%d, ILP=%d, N=%d, 2032 MHz boost) ===\n",
           BLOCKS, ILP, N_ITERS);
    printf("op           wall_ms   Glane/s   %%SoL_(1/4cy/SMSP=9.62G)\n");

    auto run = [&](const char* name, int op) {
        if (op == 0) mufu_peak<0, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 1) mufu_peak<1, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 2) mufu_peak<2, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 3) mufu_peak<3, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 4) mufu_peak<4, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 5) mufu_peak<5, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 6) mufu_peak<6, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        cudaError_t e = cudaDeviceSynchronize();
        if (e) { printf("%-12s FAIL (%s)\n", name, cudaGetErrorString(e)); cudaGetLastError(); return; }

        int RUNS = 3;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            if (op == 0) mufu_peak<0, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 1) mufu_peak<1, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 2) mufu_peak<2, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 3) mufu_peak<3, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 4) mufu_peak<4, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 5) mufu_peak<5, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 6) mufu_peak<6, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        double total = (double)BLOCKS * 128 * ILP * N_ITERS;
        double glane = total / (avg_ms / 1e3) / 1e9;
        printf("%-12s %7.3f   %7.2f   %.1f%%\n", name, avg_ms, glane, glane / THEO_GLANE * 100);
    };

    run("ex2",   0);
    run("lg2",   1);
    run("rcp",   2);
    run("rsqrt", 3);
    run("sqrt",  4);
    run("sin",   5);
    run("cos",   6);

    printf("\nMUFU theoretical (1/4cy/SMSP): %.2f Glane/s = %.2f Gops at 2032 MHz boost\n",
           THEO_GLANE, THEO_GLANE);

    cudaFree(d_out);
    return 0;
}
