// Test different TMEM ld/st shapes and widths.
// Shape: 16x64b vs 32x32b vs 16x32b
// Width: x1, x2, x4, x8, x16, x32, x64, x128

#ifndef OP
#define OP 0
#endif

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

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    unsigned acc = 0;

    #pragma unroll 1
    for (int i = 0; i < 1000; i++) {
#if OP == 0
        // ld 16x64b.x1 (1 elem per lane, 32 lanes × 8B = 256 B per warp inst)
        unsigned r;
        asm volatile("tcgen05.ld.sync.aligned.16x64b.x1.b32 {%0}, [%1];" : "=r"(r) : "r"(tmem_addr));
        acc ^= r;
#elif OP == 1
        // ld 16x64b.x2 (2 elem per lane = 512 B per warp inst)
        unsigned r0, r1;
        asm volatile("tcgen05.ld.sync.aligned.16x64b.x2.b32 {%0,%1}, [%2];"
            : "=r"(r0),"=r"(r1) : "r"(tmem_addr));
        acc ^= r0 ^ r1;
#elif OP == 2
        // ld 16x64b.x4
        unsigned r[4];
        asm volatile("tcgen05.ld.sync.aligned.16x64b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(tmem_addr));
        for (int j = 0; j < 4; j++) acc ^= r[j];
#elif OP == 3
        // ld 16x64b.x8
        unsigned r[8];
        asm volatile("tcgen05.ld.sync.aligned.16x64b.x8.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
              "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(tmem_addr));
        for (int j = 0; j < 8; j++) acc ^= r[j];
#elif OP == 4
        // ld 16x64b.x16
        unsigned r[16];
        asm volatile("tcgen05.ld.sync.aligned.16x64b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
              "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
              "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
              "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]) : "r"(tmem_addr));
        for (int j = 0; j < 16; j++) acc ^= r[j];
#elif OP == 5
        // ld 32x32b.x32 (different shape)
        unsigned r[32];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
              "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
              "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
              "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
              "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),
              "=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
              "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),
              "=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
            : "r"(tmem_addr));
        for (int j = 0; j < 32; j++) acc ^= r[j];
#endif
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;");
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem_addr));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned*)C)[2] = acc;
    }
}
