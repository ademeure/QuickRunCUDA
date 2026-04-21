// V49: Dual-pipe parallelism — does FFMA + LOP3 run concurrently at full rate?
// Theoretical: separate FMA + INT pipes should overlap, giving ~25 + 19 = 44 Glane/s combined
// If shared issue rate: limited to ~25-30

#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int OP, int ILP, int N_ITERS>
__global__ __launch_bounds__(128, 2)
void mix_pipe(unsigned* out) {
    int tid = threadIdx.x;
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
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            if (OP == 0) {
                // Solo FFMA
                asm volatile("fma.rn.f32 %0, %0, 0f3FC00000, 0f3F000000;" : "+f"(f[k]));
            } else if (OP == 1) {
                // Solo LOP3
                asm volatile("lop3.b32 %0, %0, 0xa5a5a5a5, 0x12345678, 0x96;" : "+r"(u[k]));
            } else if (OP == 2) {
                // Dual: FFMA + LOP3
                asm volatile("fma.rn.f32 %0, %0, 0f3FC00000, 0f3F000000;" : "+f"(f[k]));
                asm volatile("lop3.b32 %0, %0, 0xa5a5a5a5, 0x12345678, 0x96;" : "+r"(u[k]));
            } else if (OP == 3) {
                // Dual: FFMA + IADD3 (same FMA pipe — should NOT overlap)
                asm volatile("fma.rn.f32 %0, %0, 0f3FC00000, 0f3F000000;" : "+f"(f[k]));
                asm volatile("add.u32 %0, %0, 7;" : "+r"(u[k]));
            } else if (OP == 4) {
                // Dual: FFMA + PRMT
                asm volatile("fma.rn.f32 %0, %0, 0f3FC00000, 0f3F000000;" : "+f"(f[k]));
                asm volatile("prmt.b32 %0, %0, 0xa5a5a5a5, 0x6420;" : "+r"(u[k]));
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

    printf("=== V49 Dual-pipe overlap test ===\n");
    printf("Solo: 1 op type per slot. Dual: 2 op types per slot.\n");
    printf("If different pipes overlap, dual ≈ max(solo_a, solo_b).\n");
    printf("If shared pipe, dual ≈ solo_a + solo_b.\n\n");
    printf("Pattern              wall_ms   Glane/s_total  ratio_to_FFMA\n");

    auto run = [&](const char* lbl, int op, int slots_per_iter) {
        if (op == 0) mix_pipe<0, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 1) mix_pipe<1, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 2) mix_pipe<2, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 3) mix_pipe<3, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 4) mix_pipe<4, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        cudaError_t e = cudaDeviceSynchronize();
        if (e) { printf("%-22s FAIL\n", lbl); cudaGetLastError(); return 0.0; }

        int RUNS = 3;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            if (op == 0) mix_pipe<0, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 1) mix_pipe<1, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 2) mix_pipe<2, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 3) mix_pipe<3, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 4) mix_pipe<4, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        // total ops = blocks × threads × ILP × N × slots_per_iter
        double total = (double)BLOCKS * 128 * ILP * N_ITERS * slots_per_iter;
        double glane = total / (avg_ms / 1e3) / 1e9;
        return (double)glane;
    };

    double ffma = run("FFMA solo", 0, 1);
    printf("FFMA solo            %.3f   %.2f          1.00× (FFMA baseline)\n",
           BLOCKS * 128.0 * ILP * N_ITERS / ffma / 1e9 * 1000.0, ffma);

    double lop3 = run("LOP3 solo", 1, 1);
    printf("LOP3 solo            %.3f   %.2f          %.2f×\n", 0.0, lop3, lop3/ffma);

    double dual_ffma_lop3 = run("FFMA+LOP3 dual", 2, 2);
    printf("FFMA + LOP3 dual     %.3f   %.2f          %.2f× (sum=%.2f, max=%.2f)\n",
           0.0, dual_ffma_lop3, dual_ffma_lop3/ffma, ffma + lop3, fmax(ffma, lop3));

    double dual_ffma_iadd = run("FFMA+IADD3 dual", 3, 2);
    printf("FFMA + IADD3 dual    %.3f   %.2f          %.2f× (both FMA pipe?)\n",
           0.0, dual_ffma_iadd, dual_ffma_iadd/ffma);

    double dual_ffma_prmt = run("FFMA+PRMT dual", 4, 2);
    printf("FFMA + PRMT dual     %.3f   %.2f          %.2f×\n",
           0.0, dual_ffma_prmt, dual_ffma_prmt/ffma);

    printf("\nInterpretation:\n");
    printf("  Pure overlap (different pipes): dual_throughput = max(solo_a, solo_b)\n");
    printf("  Shared pipe: dual_throughput < (1/(1/a + 1/b)) = harmonic mean\n");

    cudaFree(d_out);
    return 0;
}
