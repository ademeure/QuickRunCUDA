// V50: Warp-specialized FFMA vs LOP3 — verifies V49 dual-issue conflict
// V49: same warp doing FFMA+LOP3 → 55% efficiency
// V50: split: half warps FFMA, half LOP3 → if pipes truly separate, both at solo rate

#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int OP, int ILP, int N_ITERS>
__global__ __launch_bounds__(256, 2)
void warp_split(unsigned* out) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    float f[8];
    unsigned u[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        f[k] = (float)(tid + k);
        u[k] = tid * 7 + k * 13;
    }

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        if (OP == 0) {
            // All warps do FFMA
            #pragma unroll
            for (int k = 0; k < ILP; k++) {
                asm volatile("fma.rn.f32 %0, %0, 0f3FC00000, 0f3F000000;" : "+f"(f[k]));
            }
        } else if (OP == 1) {
            // All warps do LOP3
            #pragma unroll
            for (int k = 0; k < ILP; k++) {
                asm volatile("lop3.b32 %0, %0, 0xa5a5a5a5, 0x12345678, 0x96;" : "+r"(u[k]));
            }
        } else if (OP == 2) {
            // Warps 0-3 (low half) FFMA, warps 4-7 (high half) LOP3
            if (warp_id < 4) {
                #pragma unroll
                for (int k = 0; k < ILP; k++) {
                    asm volatile("fma.rn.f32 %0, %0, 0f3FC00000, 0f3F000000;" : "+f"(f[k]));
                }
            } else {
                #pragma unroll
                for (int k = 0; k < ILP; k++) {
                    asm volatile("lop3.b32 %0, %0, 0xa5a5a5a5, 0x12345678, 0x96;" : "+r"(u[k]));
                }
            }
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (tid == 0) {
        unsigned acc = 0;
        for (int k = 0; k < ILP; k++) acc ^= u[k] ^ __float_as_int(f[k]);
        out[blockIdx.x] = acc + (unsigned)(t1 - t0);
    }
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    unsigned* d_out;
    CK(cudaMalloc(&d_out, 4 * prop.multiProcessorCount * 4));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_ITERS = 5000;
    const int BLOCKS = 1184;
    const int ILP = 8;
    const int THREADS = 256;  // 8 warps

    printf("=== V50 Warp-specialized vs unified pipe ===\n");
    printf("256 thr × 1184 blocks = 8 warps/CTA × 2 CTAs/SM = 16 warps/SM\n\n");
    printf("Pattern                    wall_ms   total_ops   Glane/s\n");

    auto run = [&](const char* lbl, int op) {
        if (op == 0) warp_split<0, ILP, N_ITERS><<<BLOCKS, THREADS>>>(d_out);
        if (op == 1) warp_split<1, ILP, N_ITERS><<<BLOCKS, THREADS>>>(d_out);
        if (op == 2) warp_split<2, ILP, N_ITERS><<<BLOCKS, THREADS>>>(d_out);
        cudaError_t e = cudaDeviceSynchronize();
        if (e) { printf("%-22s FAIL\n", lbl); cudaGetLastError(); return 0.0; }

        int RUNS = 3;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            if (op == 0) warp_split<0, ILP, N_ITERS><<<BLOCKS, THREADS>>>(d_out);
            if (op == 1) warp_split<1, ILP, N_ITERS><<<BLOCKS, THREADS>>>(d_out);
            if (op == 2) warp_split<2, ILP, N_ITERS><<<BLOCKS, THREADS>>>(d_out);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        // For OP=0,1: each thread does ILP ops per iter
        // For OP=2: each thread does ILP ops per iter (one type)
        double total = (double)BLOCKS * THREADS * ILP * N_ITERS;
        double glane = total / (avg_ms / 1e3) / 1e9;
        return glane;
    };

    double f = run("All FFMA", 0);
    printf("All FFMA (8 warps)         %.3f       %.0fM      %.2f\n", 0.0, BLOCKS*THREADS*ILP*N_ITERS/1e6, f);

    double l = run("All LOP3", 1);
    printf("All LOP3 (8 warps)         %.3f       %.0fM      %.2f\n", 0.0, BLOCKS*THREADS*ILP*N_ITERS/1e6, l);

    double s = run("4w FFMA + 4w LOP3", 2);
    printf("4w FFMA + 4w LOP3 split    %.3f       %.0fM      %.2f (=%.0f FFMA + %.0f LOP3)\n",
           0.0, BLOCKS*THREADS*ILP*N_ITERS/1e6, s, s/2, s/2);

    printf("\nIf pipes are TRULY separate per-SMSP:\n");
    printf("  All FFMA: %.0f Glane/s (1 inst/cy/SMSP × 4 SMSPs × 32 lanes × 148 SMs × 2.032 GHz)\n", f);
    printf("  All LOP3: %.0f Glane/s (same theoretical, slower IRL)\n", l);
    printf("  Split: each half-warp gets full pipe → %.0f FFMA + %.0f LOP3 = %.0f total\n",
           f/2, l/2, f/2 + l/2);
    printf("  Measured split: %.0f total (%.0f%% of expected separate pipes)\n",
           s, s / (f/2 + l/2) * 100);

    cudaFree(d_out);
    return 0;
}
