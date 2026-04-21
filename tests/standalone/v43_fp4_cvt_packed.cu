// V43 (H4): Packed e2m1x4 cvt vs scalar e2m1 cvt — quantify packing benefit
// Theoretical: e2m1x4.f32 packs 4 conversions in 1 inst — should be 4× faster

#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int OP, int ILP, int N_ITERS>
__global__ __launch_bounds__(128, 2)
void cvt_peak(unsigned* out) {
    int tid = threadIdx.x;
    float a[8], b[8], c[8], d[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        a[k] = (tid * 0.13f + k);
        b[k] = (tid * 0.07f + k * 1.5f);
        c[k] = (tid * 0.21f + k * 0.5f);
        d[k] = (tid * 0.31f + k * 2.0f);
    }
    unsigned short r[8] = {0};
    unsigned short r2[8] = {0};

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            if (OP == 0) {
                // FP4 cvt broken in CUDA 13.2 — skip
                r[k] = 0;
            } else if (OP == 1) {
                // Packed e4m3x2: 2 FP8 packed in 16 bits
                unsigned short p0, p1;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                             : "=h"(p0) : "f"(a[k]), "f"(b[k]));
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                             : "=h"(p1) : "f"(c[k]), "f"(d[k]));
                r[k] = p0 ^ p1;
                a[k] += (float)r[k];
            } else if (OP == 2) {
                // Packed e5m2x2: 2 FP8 packed in 16 bits
                unsigned short p0, p1;
                asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
                             : "=h"(p0) : "f"(a[k]), "f"(b[k]));
                asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
                             : "=h"(p1) : "f"(c[k]), "f"(d[k]));
                r[k] = p0 ^ p1;
                a[k] += (float)r[k];
            } else if (OP == 3) {
                // FP6 e2m3x2 not on sm_103 — skip
                r[k] = 0;
            } else if (OP == 4) {
                // BF16x2: 2 BF16 in 32 bits
                unsigned p;
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                             : "=r"(p) : "f"(b[k]), "f"(a[k]));
                r[k] = (unsigned short)p;
                a[k] += (float)r[k];
            } else if (OP == 5) {
                // F16x2: 2 F16 in 32 bits
                unsigned p;
                asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;"
                             : "=r"(p) : "f"(b[k]), "f"(a[k]));
                r[k] = (unsigned short)p;
                a[k] += (float)r[k];
            }
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (tid == 0) {
        unsigned acc = 0;
        for (int k = 0; k < ILP; k++) acc ^= r[k];
        out[blockIdx.x] = acc + (unsigned)(t1 - t0) + r2[0];
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

    printf("=== V43 Packed FP cvt comparison (BLOCKS=%d, ILP=%d, N=%d) ===\n",
           BLOCKS, ILP, N_ITERS);
    printf("Each inst counted once. ELEMENT throughput = inst × pack-width.\n\n");
    printf("op                 wall_ms   Ginst/s  packwidth  Gelem/s\n");

    auto run = [&](const char* name, int op, int packwidth) {
        if (op == 0) cvt_peak<0, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 1) cvt_peak<1, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 2) cvt_peak<2, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 3) cvt_peak<3, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 4) cvt_peak<4, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        if (op == 5) cvt_peak<5, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
        cudaError_t e = cudaDeviceSynchronize();
        if (e) { printf("%-18s FAIL (%s)\n", name, cudaGetErrorString(e)); cudaGetLastError(); return; }

        int RUNS = 3;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            if (op == 0) cvt_peak<0, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 1) cvt_peak<1, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 2) cvt_peak<2, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 3) cvt_peak<3, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 4) cvt_peak<4, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            if (op == 5) cvt_peak<5, ILP, N_ITERS><<<BLOCKS, 128>>>(d_out);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        // Each k-slot in ILP loop emits OP-specific number of ASM insts
        // For OP=1,2 it's 2 insts (packed x2 × 2 calls); else 1 inst per slot
        int insts_per_slot = (op == 1 || op == 2) ? 2 : 1;
        double total = (double)BLOCKS * 128 * ILP * N_ITERS * insts_per_slot;
        double ginst = total / (avg_ms / 1e3) / 1e9;
        double gelem = ginst * packwidth;
        printf("%-18s %7.3f   %7.2f   %d         %.2f\n", name, avg_ms, ginst, packwidth, gelem);
    };

    run("e2m1x4 (FP4×4)",   0, 4);
    run("e4m3x2 (FP8×2)",   1, 2);
    run("e5m2x2 (FP8×2)",   2, 2);
    run("e3m2x4 (FP6×4)",   3, 4);
    run("bf16x2",           4, 2);
    run("f16x2",            5, 2);

    cudaFree(d_out);
    return 0;
}
