// HBM READ ninja v2: combine small per-warp burst with ILP via 2-4 independent chains
#include <cuda_runtime.h>
#include <cstdio>

template <int IT, int CHAINS>
__launch_bounds__(256, 8) __global__ void r_ilp(const int *data, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    const int *base = data + warp_id * (IT * CHAINS * 32 * 8);
    int acc = 0;
    #pragma unroll
    for (int it = 0; it < IT; it++) {
        int regs[CHAINS][8];
        #pragma unroll
        for (int c = 0; c < CHAINS; c++) {
            const int *p = base + ((it * CHAINS + c) * 32 + lane) * 8;
            asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(regs[c][0]),"=r"(regs[c][1]),"=r"(regs[c][2]),"=r"(regs[c][3]),
                  "=r"(regs[c][4]),"=r"(regs[c][5]),"=r"(regs[c][6]),"=r"(regs[c][7])
                : "l"(p));
        }
        #pragma unroll
        for (int c = 0; c < CHAINS; c++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) acc ^= regs[c][j];
        }
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
    else if (tid == 0) out[0] = acc;
}

template <int IT, int CHAINS>
double bench(const int *d_data, int *d_out, size_t bytes, cudaEvent_t e0, cudaEvent_t e1) {
    int per_warp_bytes = IT * CHAINS * 32 * 8 * 4;
    int blocks = bytes / (256 * per_warp_bytes / 32);
    if (blocks < 1) return 0;
    for (int i = 0; i < 3; i++) r_ilp<IT, CHAINS><<<blocks, 256>>>(d_data, d_out);
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(e0);
        r_ilp<IT, CHAINS><<<blocks, 256>>>(d_data, d_out);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    return bytes / (best/1000) / 1e9;
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 4ull * 1024 * 1024 * 1024;
    int *d_data; cudaMalloc(&d_data, bytes); cudaMemset(d_data, 0xab, bytes);
    int *d_out; cudaMalloc(&d_out, 1<<20);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("# IT * CHAINS = effective ILP within warp\n");
    printf("# IT  CHAINS  per-warp(B)  GB/s    %% spec\n");
    #define R(IT, C) do { \
        double gbs = bench<IT, C>(d_data, d_out, bytes, e0, e1); \
        printf("  %3d  %3d  %8d   %6.1f  %.1f%%\n", IT, C, IT*C*32*8*4, gbs, gbs/7672*100); \
    } while(0)
    R(1, 1); R(1, 2); R(1, 4); R(1, 8);
    R(2, 1); R(2, 2); R(2, 4); R(2, 8);
    R(4, 1); R(4, 2); R(4, 4);
    R(8, 1); R(8, 2);
    return 0;
}
