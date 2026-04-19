// S2 v2: tcgen05 with proper warp-sync alloc + relinquish_alloc_permit
//
// Key insight: .sync.aligned suffix means ALL 32 threads of the warp must
// execute the instruction. Previous attempt had single-thread guard which
// caused the warp-wide sync to hang.
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(32, 1) __global__ void k_tcgen05_alloc_warp() {
    __shared__ alignas(16) unsigned tmem_addr;
    if (threadIdx.x == 0) tmem_addr = 0;
    __syncthreads();

    // ALL 32 lanes of warp 0 execute the alloc (warp-aligned)
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_addr))
    );

    // Relinquish allocation permit so other warps could alloc (we don't but spec requires)
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");

    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("tcgen05 TMEM addr: 0x%x\n", tmem_addr);
    }
    __syncthreads();

    // Dealloc — warp-wide
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"
        :: "r"(tmem_addr)
    );
}

int main() {
    cudaSetDevice(0);
    k_tcgen05_alloc_warp<<<1, 32>>>();
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(e));
        return 1;
    }
    printf("# tcgen05 alloc/dealloc completed without error\n");
    return 0;
}
