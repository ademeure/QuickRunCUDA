// S2 attempt: minimal tcgen05 PTX — just alloc+dealloc to verify it parses
// PTX 8.7+ for tcgen05 instructions; need sm_103a or sm_100a
#include <cuda_runtime.h>
#include <cstdio>

__global__ void k_tcgen05_alloc_only() {
    __shared__ alignas(16) unsigned tmem_addr;
    if (threadIdx.x == 0) tmem_addr = 0;
    __syncthreads();
    // tcgen05.alloc allocates 32 cols (smallest unit) of TMEM, returns pointer
    // Syntax: tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_ptr], num_cols;
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n"
            :: "r"((unsigned)__cvta_generic_to_shared(&tmem_addr))
        );
    }
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("tcgen05 TMEM addr: 0x%x\n", tmem_addr);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"
            :: "r"(tmem_addr)
        );
    }
}

int main() {
    cudaSetDevice(0);
    k_tcgen05_alloc_only<<<1, 32>>>();
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(e));
        return 1;
    }
    printf("# tcgen05 alloc/dealloc completed without error\n");
    return 0;
}
