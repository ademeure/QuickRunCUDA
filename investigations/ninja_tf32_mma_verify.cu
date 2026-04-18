// TF32 mma.sync peak verify (catalog 288 TFLOPS MED conf)
//
// mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32:
//   m=16, n=8, k=8 → 16*8*8*2 = 2048 FLOPs per warp instruction
//   At 1 mma per 4 cy per SMSP (warp-sync legacy):
//     Per SMSP: 2048/4 = 512 FLOPs/cy
//     Total: 4 * 148 * 512 * 2.032e9 = 615 TFLOPS theor (similar to BF16!)
//   But TF32 K=8 vs BF16 K=16 → mma instr count doubles for same K-product
//   So TF32 ~ 1/2 of BF16's effective rate
#include <cuda_runtime.h>
#include <cstdio>

template <int ILP>
__launch_bounds__(1024, 1) __global__ void tf32_loop(float *out, int N) {
    unsigned warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned a0=0x3F800000u, a1=0x3F800000u, a2=0x3F800000u, a3=0x3F800000u;  // 1.0f as TF32 = same bits as FP32
    unsigned b0=0x3F800000u, b1=0x3F800000u;
    float regs[ILP * 4];
    #pragma unroll
    for (int i = 0; i < ILP * 4; i++) regs[i] = (float)(warp_id * (i+1)) * 1e-30f;

    #pragma unroll 1
    for (int o = 0; o < N; o++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                : "+f"(regs[j*4+0]),"+f"(regs[j*4+1]),"+f"(regs[j*4+2]),"+f"(regs[j*4+3])
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
    }
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < ILP * 4; i++) sum += regs[i];
    if (__float_as_int(sum) == 0xdeadbeef) out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto run = [&](int ilp_label, void(*kfn)(float*, int), int blocks_per_sm) {
        int blocks = 148 * blocks_per_sm;
        int bs = 1024;
        int N = 5000;
        for (int i = 0; i < 3; i++) kfn<<<blocks, bs>>>(d_out, N);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR ILP=%d: %s\n", ilp_label, cudaGetErrorString(cudaGetLastError())); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, bs>>>(d_out, N);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long total_mma = (long)blocks * (bs/32) * N * ilp_label;
        long total_flops = total_mma * 2048L;  // m16 n8 k8 fma=2 = 2048
        double tflops = total_flops / (best/1000.0) / 1e12;
        printf("  ILP=%-3d  bs=%d blocks=%d (%dw/SM)  %.4f ms = %.1f TFLOPS\n", ilp_label, bs, blocks, blocks_per_sm * (bs/32), best, tflops);
    };

    printf("# TF32 mma.sync peak (m16n8k8)\n");
    run(1, tf32_loop<1>, 1);
    run(2, tf32_loop<2>, 1);
    run(4, tf32_loop<4>, 1);
    run(8, tf32_loop<8>, 1);
    run(2, tf32_loop<2>, 2);  // 64w/SM
    run(2, tf32_loop<2>, 4);
    return 0;
}
