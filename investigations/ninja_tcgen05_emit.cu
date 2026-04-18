// Just emit tcgen05 instructions and verify they assemble for sm_103a
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void try_alloc_dealloc() {
    __shared__ unsigned tmem_ptr;
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n"
            :: "l"(&tmem_ptr));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // Free what we just allocated
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"
            :: "r"(tmem_ptr));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }
}

// Try forming a tcgen05.mma.kind::mxf4 instruction
// Per PTX 8.7+: tcgen05.mma.cta_group::1.kind::mxf4.collector::a::fill d, a, b, instr_desc, scale_a, scale_b, scale_d_imm
// This needs descriptors for a, b in SMEM. Simplified to just verify assembly:
extern "C" __global__ void try_mma_mxf4() {
    __shared__ unsigned tmem_ptr;
    __shared__ unsigned long long a_smem_desc;
    __shared__ unsigned long long b_smem_desc;
    __shared__ unsigned a_scale_tmem;
    __shared__ unsigned b_scale_tmem;
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;\n"
            :: "l"(&tmem_ptr));
        a_smem_desc = 0; b_smem_desc = 0; a_scale_tmem = 0; b_scale_tmem = 0;
        unsigned instr_desc = 0;  // would encode m=128, n=128, k=96 + types
        // Try the actual mma instruction
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.collector::a::fill "
            "[%0], %1, %2, %3, [%4], [%5], 0;\n"
            :: "r"(tmem_ptr), "l"(a_smem_desc), "l"(b_smem_desc), "r"(instr_desc),
               "r"(a_scale_tmem), "r"(b_scale_tmem));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;\n"
            :: "r"(tmem_ptr));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }
}

int main() {
    cudaSetDevice(0);
    try_alloc_dealloc<<<1, 32>>>();
    cudaError_t err = cudaDeviceSynchronize();
    printf("try_alloc_dealloc: %s\n", cudaGetErrorString(err));

    try_mma_mxf4<<<1, 32>>>();
    err = cudaDeviceSynchronize();
    printf("try_mma_mxf4: %s\n", cudaGetErrorString(err));
    return 0;
}
