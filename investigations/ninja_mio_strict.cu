// MIO unification re-verify with strict anti-DCE
#include <cuda_runtime.h>
#include <cstdio>
constexpr int SMEM_INTS = 32 * 1024;
constexpr int N_INNER = 64;

__launch_bounds__(1024, 1) __global__ void k_sts_only(unsigned *out, int N) {
    __shared__ int smem[SMEM_INTS];
    volatile int *vsmem = smem;
    int slot = (threadIdx.x>>5)*32 + (threadIdx.x&31);
    int v = threadIdx.x;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            vsmem[(slot + j*1024) & (SMEM_INTS-1)] = v + i + j;
        }
    }
    out[blockIdx.x*1024 + threadIdx.x] = v;  // strict: always-write v (constant per-thread)
}
__launch_bounds__(1024, 1) __global__ void k_shfl_only(unsigned *out, int N) {
    int v = threadIdx.x;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            v = __shfl_xor_sync(0xFFFFFFFF, v, j & 31) + i;
        }
    }
    out[blockIdx.x*1024 + threadIdx.x] = v;  // strict
}
__launch_bounds__(1024, 1) __global__ void k_sts_shfl(unsigned *out, int N) {
    __shared__ int smem[SMEM_INTS];
    volatile int *vsmem = smem;
    int slot = (threadIdx.x>>5)*32 + (threadIdx.x&31);
    int v = threadIdx.x;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            int s = (slot + j*1024) & (SMEM_INTS-1);
            vsmem[s] = v + i + j;
            v = __shfl_xor_sync(0xFFFFFFFF, v, j & 31) + i;
        }
    }
    out[blockIdx.x*1024 + threadIdx.x] = v;  // strict
}

int main() {
    cudaSetDevice(0);
    unsigned *out; cudaMalloc(&out, 148*1024*sizeof(unsigned));
    int N = 200;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto bench = [&](const char* name, auto launch) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        printf("  %-15s  %.3f ms\n", name, best);
    };
    bench("STS only",  [&](){k_sts_only<<<148, 1024>>>(out, N);});
    bench("SHFL only", [&](){k_shfl_only<<<148, 1024>>>(out, N);});
    bench("STS+SHFL",  [&](){k_sts_shfl<<<148, 1024>>>(out, N);});
    return 0;
}
