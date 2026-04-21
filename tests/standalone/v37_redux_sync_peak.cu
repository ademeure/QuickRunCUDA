// V37: redux.sync peak throughput — warp-level reduction SoL
// 10-rule rigor:
// Theoretical: redux.sync.max is 1 warp-inst per cycle per SMSP = 1 inst/cy/SMSP.
// 148 SMs × 4 SMSPs × 2032 MHz × 1 inst/cy = 1202 Ginst/s.
// Each warp-inst reduces 32 values → 38.5 Gvalues/s/SMSP × 4 × 148 = ~22.8 Telements/s.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

// Kernel: 8 independent redux chains in parallel (ILP)
template<int N_REDUX, int N_ITERS>
__global__ __launch_bounds__(128, 2)
void redux_peak(unsigned* out) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    unsigned v[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) v[k] = (lane * 7 + k * 13) ^ (blockIdx.x * 37);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned r;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;"
                         : "=r"(r) : "r"(v[k]));
            v[k] = r + 1;  // independent chain per slot
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (tid == 0) {
        unsigned acc = 0;
        for (int k = 0; k < 8; k++) acc ^= v[k];
        out[blockIdx.x] = acc + (unsigned)(t1 - t0);
    }
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;

    unsigned* d_out;
    CK(cudaMalloc(&d_out, 4 * sm_count * 2));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_REDUX = 8;  // ILP=8 chains
    const int N_ITERS = 10000;

    printf("=== V37 redux.sync.max peak throughput ===\n");
    printf("Kernel: %d unrolled × %d iters × 4 warps/CTA × N blocks\n\n", N_REDUX, N_ITERS);
    printf("blocks  wall_ms   redux_inst  Ginst/s  Gelement/s   %%SoL\n");

    const double THEO_GINST = 148.0 * 4 * 2.032;  // inst/cy/SMSP × SMSPs × SMs × GHz
    const double THEO_GELEM = THEO_GINST * 32;  // 32 lanes per warp

    for (int blocks : {148, 296, 592, 1184}) {
        // warmup
        redux_peak<N_REDUX, N_ITERS><<<blocks, 128>>>(d_out);
        cudaDeviceSynchronize();

        int RUNS = 3;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            redux_peak<N_REDUX, N_ITERS><<<blocks, 128>>>(d_out);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;

        double total_warp_inst = (double)blocks * 4 * 8 * N_ITERS;  // 4 warps/CTA × 8 ILP chains
        double ginst = total_warp_inst / (avg_ms / 1e3) / 1e9;
        double gelem = ginst * 32;
        printf("%4d    %7.3f   %10.2fG   %7.2f    %8.2f     %.1f%%\n",
               blocks, avg_ms, total_warp_inst / 1e9, ginst, gelem, ginst / THEO_GINST * 100);
    }
    printf("\nTheoretical: %.2f Ginst/s = %.2f Gelements/s at 2032 MHz boost\n", THEO_GINST, THEO_GELEM);

    cudaFree(d_out);
    return 0;
}
