extern "C" __global__ __launch_bounds__(128, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    __shared__ __align__(4) unsigned tmem_slot;
    if (threadIdx.x == 0) tmem_slot = 0xFFFFFFFF;
    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;"
            :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    }
    __syncthreads();
    unsigned tmem_addr = tmem_slot;
    unsigned acc = 0;
    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < 1000; i++) {
#if OP == 0
        // 16x256b.x1 — 4 regs per lane (256-bit / 32 lanes spread)
        unsigned r[4];
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(tmem_addr));
        for (int j = 0; j < 4; j++) acc ^= r[j];
#elif OP == 1
        // 16x128b.x1 — 2 regs per lane
        unsigned r[2];
        asm volatile("tcgen05.ld.sync.aligned.16x128b.x1.b32 {%0,%1}, [%2];"
            : "=r"(r[0]),"=r"(r[1]) : "r"(tmem_addr));
        acc ^= r[0] ^ r[1];
#elif OP == 2
        // 16x32bx2.x32 — explore wider
        unsigned r[16];
        asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16], 0;"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
              "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
              "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
              "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
            : "r"(tmem_addr));
        for (int j = 0; j < 16; j++) acc ^= r[j];
#elif OP == 3
        // 16x256b.x4 — 16 regs per lane
        unsigned r[16];
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
              "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
              "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
              "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
            : "r"(tmem_addr));
        for (int j = 0; j < 16; j++) acc ^= r[j];
#endif
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;");
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem_addr));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
    if (threadIdx.x == 0) ((unsigned long long*)C)[0] = t1 - t0;
}
