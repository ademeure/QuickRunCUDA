// Minimal tcgen05 smoke test — single warp alloc/dealloc
extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ unsigned int tmem_addr;
    if (threadIdx.x >= 32) return;

    // tcgen05.alloc requires whole warp to call
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
        :: "r"((unsigned int)__cvta_generic_to_shared(&tmem_addr))
    );
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
    );
    __syncwarp();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("tmem_addr = 0x%x\n", tmem_addr);
    }

    __syncwarp();
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
        :: "r"(tmem_addr)
    );

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("dealloc OK\n");
    }
}
