// H7: TMEM idle power - allocate TMEM but don't access; measure power
// MODE 0: no TMEM, just spin
// MODE 1: alloc TMEM 256 columns, spin (no access)
// MODE 2: alloc TMEM 512 columns (max), spin
// Compare to baseline spinning kernel
#include <cuda_runtime.h>
#include <cstdio>

__global__ __launch_bounds__(32, 1)
void spin_no_tmem(unsigned long long delay) {
    if (threadIdx.x == 0) {
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t0 < delay);
    }
}

__global__ __launch_bounds__(32, 1)
void spin_with_tmem_256(unsigned long long delay) {
    __shared__ __align__(4) unsigned tmem_slot;
    if (threadIdx.x == 0) tmem_slot = 0xFFFFFFFF;
    __syncthreads();

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    __syncthreads();
    unsigned tmem_addr = tmem_slot;

    if (threadIdx.x == 0) {
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t0 < delay);
    }
    __syncthreads();

    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem_addr));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}

__global__ __launch_bounds__(32, 1)
void spin_with_tmem_512(unsigned long long delay) {
    __shared__ __align__(4) unsigned tmem_slot;
    if (threadIdx.x == 0) tmem_slot = 0xFFFFFFFF;
    __syncthreads();

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    __syncthreads();
    unsigned tmem_addr = tmem_slot;

    if (threadIdx.x == 0) {
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t0 < delay);
    }
    __syncthreads();

    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}

int main(int argc, char** argv) {
    int mode = (argc > 1) ? atoi(argv[1]) : 0;
    unsigned long long delay = 4500000000ULL;  // 3 sec at 1500 MHz
    int blocks = 148;  // 1 block per SM (each SM allocates own TMEM)

    if (mode == 0) spin_no_tmem<<<blocks, 32>>>(delay);
    else if (mode == 1) spin_with_tmem_256<<<blocks, 32>>>(delay);
    else if (mode == 2) spin_with_tmem_512<<<blocks, 32>>>(delay);
    cudaDeviceSynchronize();
    return 0;
}
