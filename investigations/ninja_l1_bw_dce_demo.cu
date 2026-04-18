#include <cuda_runtime.h>
#include <cstdio>
template <int ILP>
__launch_bounds__(256, 8) __global__ void l1_hot(uint4 *src, uint4 *sink, int N_iters) {
    int tid = threadIdx.x;
    uint4 *p = src + tid;
    uint4 acc = make_uint4(0,0,0,0);
    p[0];
    #pragma unroll 1
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            uint4 v = p[j * 256];
            acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
        }
    }
    if (acc.x == 0xdeadbeef) sink[blockIdx.x * blockDim.x + tid] = acc;
}
template <int ILP>
double bench(int blocks, int N_iters) {
    size_t ws = (size_t)ILP * 256 * 16;
    uint4 *d_src; cudaMalloc(&d_src, ws); cudaMemset(d_src, 0xab, ws);
    uint4 *d_sink; cudaMalloc(&d_sink, blocks * 256 * 16);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) l1_hot<ILP><<<blocks, 256>>>(d_src, d_sink, N_iters);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR ILP=%d: %s\n", ILP, cudaGetErrorString(cudaGetLastError())); return 0; }
    float best = 1e30f;
    for (int i = 0; i < 8; i++) {
        cudaEventRecord(e0);
        l1_hot<ILP><<<blocks, 256>>>(d_src, d_sink, N_iters);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long total = (long)blocks * 256 * N_iters * ILP * 16;
    double tbs = total / (best/1000.0) / 1e12;
    printf("  ILP=%-3d ws=%zu KB  blocks=%d N=%d  %.4f ms = %.2f TB/s\n", ILP, ws/1024, blocks, N_iters, best, tbs);
    cudaFree(d_src); cudaFree(d_sink);
    return tbs;
}
int main() {
    cudaSetDevice(0);
    int blocks = 1184;
    int N = 30000;
    bench<1>(blocks, N);
    bench<2>(blocks, N);
    bench<4>(blocks, N);
    bench<8>(blocks, N);
    bench<16>(blocks, N);
    bench<32>(blocks, N);
    bench<64>(blocks, N);
    return 0;
}
