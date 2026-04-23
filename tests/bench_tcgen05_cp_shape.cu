// Test which tcgen05.cp shapes ptxas accepts and which run successfully.
// Shape via -H "#define SHAPE \"128x128b\"" etc.
#ifndef SHAPE
#define SHAPE "128x128b"
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int u0, int u1, int u2) {
    __shared__ __align__(1024) unsigned smem[8192];
    __shared__ __align__(4) unsigned tmem_slot;

    if (threadIdx.x < 32) {
        for (int i = threadIdx.x; i < 8192; i += 32) smem[i] = 0xCAFE0000u | i;
    }
    if (threadIdx.x == 0) tmem_slot = 0xFFFFFFFFu;
    __syncthreads();

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    __syncthreads();

    unsigned tmem_addr = tmem_slot;
    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem);
    unsigned long long desc = desc_encode((unsigned long long)smem_addr)
                            | (desc_encode(16) << 16)
                            | (desc_encode(2048) << 32);

    if (threadIdx.x == 0) {
        asm volatile("tcgen05.cp.cta_group::1." SHAPE " [%0], %1;"
            :: "r"(tmem_addr), "l"(desc));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[CP-SHAPE] " SHAPE " RAN OK tmem_addr=0x%08x\n", tmem_addr);
        ((unsigned*)C)[0] = 0xC0FFEEu;
    }
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem_addr));
}
