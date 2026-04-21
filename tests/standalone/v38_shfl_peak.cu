// V38: shfl.sync peak throughput — for comparison to redux (V37)
// Theoretical: 1 shfl/cy/SMSP × 4 × 148 × 2.032 = 1202 Ginst/s = 38.5 Telements/s

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int N_ITERS>
__global__ __launch_bounds__(128, 2)
void shfl_peak(unsigned* out) {
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
            asm volatile("shfl.sync.bfly.b32 %0, %1, 16, 0x1f, 0xffffffff;"
                         : "=r"(r) : "r"(v[k]));
            v[k] = r + 1;
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

    unsigned* d_out;
    CK(cudaMalloc(&d_out, 4 * prop.multiProcessorCount * 2));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_ITERS = 10000;
    const double THEO_GINST = 148.0 * 4 * 2.032;

    printf("=== V38 shfl.sync.bfly peak throughput ===\n");
    printf("blocks  wall_ms   shfl_inst  Ginst/s  Gelement/s   %%SoL\n");

    for (int blocks : {148, 296, 592, 1184, 2368}) {
        shfl_peak<N_ITERS><<<blocks, 128>>>(d_out);
        cudaDeviceSynchronize();

        int RUNS = 3;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            shfl_peak<N_ITERS><<<blocks, 128>>>(d_out);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        double total = (double)blocks * 4 * 8 * N_ITERS;
        double ginst = total / (avg_ms / 1e3) / 1e9;
        printf("%4d    %7.3f   %10.2fG   %7.2f    %8.2f     %.1f%%\n",
               blocks, avg_ms, total / 1e9, ginst, ginst * 32, ginst / THEO_GINST * 100);
    }
    printf("\nTheoretical: %.2f Ginst/s = %.2f Gelements/s at 2032 MHz\n", THEO_GINST, THEO_GINST * 32);

    cudaFree(d_out);
    return 0;
}
