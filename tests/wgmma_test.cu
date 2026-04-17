// Try Hopper-style WGMMA (warp group matrix multiply async) on B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define ITERS 1000

// wgmma uses descriptor-based addressing for A and B in shared memory
// But for simple test, can use register inputs for A
extern "C" __global__ void k_wgmma_test(float *out) {
    // wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16
    // A: 64x16 BF16 in registers (shared layout)
    // B: 16x256 BF16 in shared memory (descriptor)
    // D: 64x256 FP32 = 64 fp32 per thread (per warp group)

    // For now try tiny version: m64n8k16 to see if syntax works
    extern __shared__ unsigned int smem[];
    int tid = threadIdx.x;
    if (tid < 1024) smem[tid] = 0x3F803F80;
    __syncthreads();

    // Attempt wgmma — needs warp group of 4 warps (128 threads)
    // Skip for now; just measure if it compiles

    if (tid == 0) out[blockIdx.x] = 1.0f;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);

    float *d_out;
    cudaMalloc(&d_out, prop.multiProcessorCount * sizeof(float));

    k_wgmma_test<<<prop.multiProcessorCount, 128, 4096>>>(d_out);
    cudaDeviceSynchronize();
    cudaError_t r = cudaGetLastError();
    printf("wgmma test launch: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));

    // Note: full wgmma syntax is complex (matrix descriptors, smem alignment, etc.)
    // For B300, the Blackwell-specific tcgen05.mma is the path to peak BF16/FP8/FP4

    cudaFree(d_out);
    return 0;
}
