// L1 BW with bulletproof anti-DCE: address depends on accumulator,
// so each iteration's load addresses depend on prior loads' values.
#include <cuda_runtime.h>
#include <cstdio>

template <int ILP>
__launch_bounds__(256, 8) __global__ void l1_hot_safe(uint4 *src, uint4 *sink, int N_iters, int mask) {
    int tid = threadIdx.x;
    uint4 acc = make_uint4(0,0,0,0);
    int idx = tid;
    #pragma unroll 1
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            uint4 v = src[(idx + j * 256) & mask];
            acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
        }
        // Make next iter depend on acc — defeats provability of DCE
        idx = ((acc.x ^ acc.y ^ acc.z ^ acc.w) & 0xff) + tid;
    }
    sink[blockIdx.x * blockDim.x + tid] = acc;  // unconditional store
}

template <int ILP>
double bench(int blocks, int N_iters) {
    int n_uint4 = ILP * 256;  // working set in uint4s
    int mask = (n_uint4 < 256) ? 255 : (n_uint4 - 1);  // assume ILP*256 is power of 2
    uint4 *d_src; cudaMalloc(&d_src, n_uint4 * 16);
    cudaMemset(d_src, 0xab, n_uint4 * 16);
    uint4 *d_sink; cudaMalloc(&d_sink, blocks * 256 * 16);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) l1_hot_safe<ILP><<<blocks, 256>>>(d_src, d_sink, N_iters, mask);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR ILP=%d: %s\n", ILP, cudaGetErrorString(cudaGetLastError())); return 0; }
    float best = 1e30f;
    for (int i = 0; i < 8; i++) {
        cudaEventRecord(e0);
        l1_hot_safe<ILP><<<blocks, 256>>>(d_src, d_sink, N_iters, mask);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long total = (long)blocks * 256 * N_iters * ILP * 16;
    double tbs = total / (best/1000.0) / 1e12;
    printf("  ILP=%-3d ws=%dKB blocks=%d  %.4f ms = %.2f TB/s\n", ILP, n_uint4*16/1024, blocks, best, tbs);
    cudaFree(d_src); cudaFree(d_sink);
    return tbs;
}

int main() {
    cudaSetDevice(0);
    int blocks = 1184;
    int N = 10000;
    bench<1>(blocks, N);
    bench<2>(blocks, N);
    bench<4>(blocks, N);
    bench<8>(blocks, N);
    bench<16>(blocks, N);
    bench<32>(blocks, N);
    return 0;
}
