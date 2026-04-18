// Pure MUFU.EX2 — use ex2.approx.ftz.f32 to skip range reduction
#include <cuda_runtime.h>
#include <cstdio>
template <int ILP>
__launch_bounds__(256, 8) __global__ void xu_pure(float *out, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float regs[ILP];
    #pragma unroll
    for (int i = 0; i < ILP; i++) regs[i] = (float)tid * 0.0001f + i * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(regs[j]));
        }
    }
    float acc = 0;
    #pragma unroll
    for (int j = 0; j < ILP; j++) acc += regs[j];
    if (acc == 0xdeadbeef) out[tid] = acc;
}
int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8;
    int N = 100000;
    auto run = [&](int ilp_label, void(*kfn)(float*, int)) {
        for (int i = 0; i < 3; i++) kfn<<<blocks, 256>>>(d_out, N);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, 256>>>(d_out, N);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * 256 * N * ilp_label;
        double gops = ops / (best/1000.0) / 1e9;
        printf("ILP=%-3d %.4f ms = %.0f Gops/s pure MUFU.EX2\n", ilp_label, best, gops);
    };
    run(1, xu_pure<1>);
    run(2, xu_pure<2>);
    run(4, xu_pure<4>);
    run(8, xu_pure<8>);
    run(16, xu_pure<16>);
    run(32, xu_pure<32>);
    return 0;
}
