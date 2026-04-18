// XU pipe (MUFU) aggregate peak throughput verification
//
// Theoretical:
//   ex2.approx.f32 = 14 cy/issue, 0.5 issue/SMSP/cy
//   aggregate: 0.5 * 4 SMSPs * 148 SMs * 2.032 GHz = 600 Gops/s
#include <cuda_runtime.h>
#include <cstdio>

template <int ILP>
__launch_bounds__(256, 8) __global__ void xu_ex2_loop(float *out, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float seed = (float)tid * 0.001f;
    float regs[ILP];
    #pragma unroll
    for (int i = 0; i < ILP; i++) regs[i] = seed + i * 0.0001f;

    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            asm volatile("ex2.approx.f32 %0, %0;\n" : "+f"(regs[j]));
        }
    }
    float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < ILP; j++) acc += regs[j];
    if (acc == 0xdeadbeef) out[tid] = acc;  // anti-DCE
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
        printf("ILP=%-3d %.4f ms = %.0f Gops/s ex2  (theoretical 600)\n", ilp_label, best, gops);
    };

    run(1,  xu_ex2_loop<1>);
    run(2,  xu_ex2_loop<2>);
    run(4,  xu_ex2_loop<4>);
    run(8,  xu_ex2_loop<8>);
    run(16, xu_ex2_loop<16>);
    run(32, xu_ex2_loop<32>);
    return 0;
}
