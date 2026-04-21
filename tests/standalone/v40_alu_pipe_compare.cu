// V40: ALU pipe peaks compared — FADD, FFMA, IADD3, LOP3, PRMT, ISETP
// All should hit 1 inst/cy/SMSP if same pipe.
// Theoretical: 1 × 4 SMSPs × 148 SMs × 2.032 GHz × 32 lanes = 38.49 Glanes/sec

#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int OP, int ILP, int N_ITERS>
__global__ __launch_bounds__(128, 2)
void alu_peak(unsigned* out) {
    int tid = threadIdx.x;
    unsigned u[8] = {tid+1u, tid+2u, tid+3u, tid+4u, tid+5u, tid+6u, tid+7u, tid+8u};
    float f[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) f[k] = (float)u[k] + 0.1f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            if (OP == 0) asm volatile("add.f32 %0, %0, 0f3FC00000;" : "+f"(f[k]));
            else if (OP == 1) asm volatile("fma.rn.f32 %0, %0, 0f3FC00000, 0f3F000000;" : "+f"(f[k]));
            else if (OP == 2) asm volatile("add.u32 %0, %0, 7;" : "+r"(u[k]));
            else if (OP == 3) asm volatile("lop3.b32 %0, %0, 0xa5a5a5a5, 0x12345678, 0x96;" : "+r"(u[k]));
            else if (OP == 4) asm volatile("prmt.b32 %0, %0, 0xdeadbeef, 0x6420;" : "+r"(u[k]));
            else if (OP == 5) {
                int p;
                asm volatile("{.reg .pred q;\nsetp.lt.s32 q, %1, 100;\nselp.s32 %0, 1, 0, q;}\n"
                             : "=r"(p) : "r"((int)u[k]));
                u[k] += p;
            }
            else if (OP == 6) asm volatile("mul.lo.u32 %0, %0, 7;" : "+r"(u[k]));
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
    const double THEO_GLANE = 148.0 * 4 * 32 * 2.032;

    printf("=== V40 ALU pipe peak compare (BLOCKS=%d, ILP=%d, N=%d) ===\n",
           BLOCKS, ILP, N_ITERS);
    printf("op           wall_ms  Glane/s   %%SoL\n");

    auto run = [&](const char* name, int op) {
        cudaError_t e;
        if (op == 0) alu_peak<0, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 1) alu_peak<1, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 2) alu_peak<2, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 3) alu_peak<3, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 4) alu_peak<4, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 5) alu_peak<5, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 6) alu_peak<6, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        e = cudaDeviceSynchronize();
        if (e) { printf("%-12s FAIL\n", name); cudaGetLastError(); return; }
        int RUNS = 3;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            if (op == 0) alu_peak<0, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 1) alu_peak<1, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 2) alu_peak<2, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 3) alu_peak<3, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 4) alu_peak<4, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 5) alu_peak<5, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 6) alu_peak<6, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        double total = (double)BLOCKS * 128 * ILP * N_ITERS;
        double glane = total / (avg_ms / 1e3) / 1e9;
        printf("%-12s %7.3f  %8.2f   %.1f%%\n", name, avg_ms, glane, glane / THEO_GLANE * 100);
    };

    run("FADD",  0);
    run("FFMA",  1);
    run("IADD3", 2);
    run("LOP3",  3);
    run("PRMT",  4);
    run("ISETP", 5);
    run("IMUL",  6);

    printf("\nTheoretical: %.2f Glanes/s (1 warp-inst/cy/SMSP × 32 lanes)\n", THEO_GLANE);

    cudaFree(d_out);
    return 0;
}
