// True L2 BW: spread accesses across full WS to defeat L1
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
__launch_bounds__(256, 4) __global__ void l2_real(uint4 *src, uint4 *sink, int N_iters, int N_uint4) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    uint4 acc = make_uint4(0,0,0,0);
    int mask = N_uint4 - 1;
    int idx = (tid + bid * 4096) & mask;  // each block starts at different offset
    #pragma unroll 1
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int addr = (idx + j * 1024) & mask;  // 1024 stride = 16 KB per j step
            uint4 v;
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(&src[addr]));
            acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
        }
        // Spread idx across whole WS via dependency on acc
        idx = (idx + 256 + (acc.x & 0xfff)) & mask;
    }
    sink[bid * blockDim.x + tid] = acc;
}
int main() {
    cudaSetDevice(0);
    // Try bigger WS that doesn't fit in L1 working sets
    for (int WS_MB : {8, 16, 32, 48, 56}) {
        int N_uint4 = WS_MB * 1024 * 1024 / 16;
        uint4 *d_src; cudaMalloc(&d_src, N_uint4 * 16);
        cudaMemset(d_src, 0xab, N_uint4 * 16);
        uint4 *d_sink; cudaMalloc(&d_sink, 4096 * 256 * 16);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        // Just measure at saturation block count
        int blocks = 1184; int N_iters = 500;
        l2_real<<<blocks, 256>>>(d_src, d_sink, 3, N_uint4);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR ws=%d: %s\n", WS_MB, cudaGetErrorString(cudaGetLastError())); cudaFree(d_src); cudaFree(d_sink); continue; }
        float best = 1e30f;
        for (int i = 0; i < 8; i++) {
            cudaEventRecord(e0);
            l2_real<<<blocks, 256>>>(d_src, d_sink, N_iters, N_uint4);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long total = (long)blocks * 256 * N_iters * 8 * 16;
        double tbs = total / (best/1000.0) / 1e12;
        printf("  WS=%2d MB blocks=%d: %.3f ms = %.2f TB/s\n", WS_MB, blocks, best, tbs);
        cudaFree(d_src); cudaFree(d_sink);
    }
    return 0;
}
