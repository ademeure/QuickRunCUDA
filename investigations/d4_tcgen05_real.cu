// D4 RIGOR: tcgen05.mma direct microbench for FP4/FP8/BF16
// Compile with: nvcc -gencode arch=compute_103a,code=sm_103a
//
// THEORETICAL: NVIDIA B300 spec dense:
//   FP4:  ~10000 TFLOPS (10 PFLOPS)
//   FP8:  ~5000 TFLOPS
//   BF16: ~2500 TFLOPS
//
// Approach: minimal tcgen05.mma loop with mbarrier wait

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// tcgen05 state — using cta_group::1 (single-CTA) for simplicity
// kind::f8f6f4 supports FP4/FP6/FP8 inputs

#ifndef ITERS
#define ITERS 256
#endif

// Single-CTA tcgen05 MMA — minimum size m=64 n=8 k=32 for kind::f8f6f4
// Output: 64x8 = 512 elements of f32 = 2048 bytes in tensor mem
// A: 64x32 of FP8 = 2048 bytes (32 cols/thread for 1-CTA layout?)
// B: 32x8 of FP8 = 256 bytes

// instr_desc layout for tcgen05.mma kind::f8f6f4:
//   bits  0:5  = M >> 4  (M/16)
//   bits  6:8  = N >> 3  (N/8)
//   bits  9:13 = (reserved/unused for some variants)
//   bits 14:17 = A_kind (FP8 e4m3 = 0, FP8 e5m2 = 1, FP6 e2m3 = 2, FP6 e3m2 = 3, FP4 e2m1 = 4)
//   bits 18:21 = B_kind
//   bits 22:31 = various flags (sparse, scaling, etc.)
// (Per PTX docs §17.5.7)

__global__ void tcgen05_smoke(float *out) {
    __shared__ unsigned tmem_addr_storage;
    __shared__ uint64_t mbar;

    if (threadIdx.x == 0) {
        // Allocate 32 columns of tensor memory
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n"
            :: "l"(&tmem_addr_storage)
        );
        // Init mbarrier with arrival count of 1 (the warp issuing the MMA)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "l"(&mbar));
        asm volatile("fence.proxy.async.shared::cta;\n" ::);
    }
    __syncthreads();

    // Just check that we got an address back
    if (threadIdx.x == 0) {
        out[0] = (float)tmem_addr_storage;
    }

    // Free tensor memory
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"
            :: "r"(tmem_addr_storage)
        );
    }
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 1024 * sizeof(float));
    cudaMemset(d_out, 0, 1024 * sizeof(float));
    tcgen05_smoke<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("# tcgen05 smoke test: tmem_addr returned = 0x%x (= %f)\n",
           (unsigned)h_out, h_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("  ERR: %s\n", cudaGetErrorString(err));
    else printf("  PASS\n");
    return 0;
}
