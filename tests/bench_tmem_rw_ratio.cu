// TMEM read/write ratio sweep — 1R:0W, 0R:1W, 1R:1W, 1R:2W, 1R:3W, 2R:1W, 3R:1W
// All 4 warps participate; each warp does same per-iter pattern.

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

    unsigned vals[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) vals[i] = (threadIdx.x << 16) | (i + seed);

    // Pre-fill TMEM
    asm volatile("tcgen05.st.sync.aligned.16x64b.x16.b32 "
        "[%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
        :: "r"(tmem_addr),
           "r"(vals[0]),"r"(vals[1]),"r"(vals[2]),"r"(vals[3]),
           "r"(vals[4]),"r"(vals[5]),"r"(vals[6]),"r"(vals[7]),
           "r"(vals[8]),"r"(vals[9]),"r"(vals[10]),"r"(vals[11]),
           "r"(vals[12]),"r"(vals[13]),"r"(vals[14]),"r"(vals[15]) : "memory");
    asm volatile("tcgen05.wait::st.sync.aligned;");

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    unsigned acc = 0;
    #pragma unroll 1
    for (int i = 0; i < 1000; i++) {
#define DO_LD do { \
    unsigned r[16]; \
    asm volatile("tcgen05.ld.sync.aligned.16x64b.x16.b32 " \
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];" \
        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]), \
          "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]), \
          "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]), \
          "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]) \
        : "r"(tmem_addr)); \
    for (int j = 0; j < 16; j++) acc ^= r[j]; \
} while(0)
#define DO_ST do { \
    asm volatile("tcgen05.st.sync.aligned.16x64b.x16.b32 " \
        "[%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" \
        :: "r"(tmem_addr), \
           "r"(vals[0]),"r"(vals[1]),"r"(vals[2]),"r"(vals[3]), \
           "r"(vals[4]),"r"(vals[5]),"r"(vals[6]),"r"(vals[7]), \
           "r"(vals[8]),"r"(vals[9]),"r"(vals[10]),"r"(vals[11]), \
           "r"(vals[12]),"r"(vals[13]),"r"(vals[14]),"r"(vals[15]) : "memory"); \
} while(0)

#if OP == 0
        DO_LD;                         // 1R
#elif OP == 1
        DO_ST;                         // 1W
#elif OP == 2
        DO_LD; DO_ST;                  // 1R + 1W
#elif OP == 3
        DO_LD; DO_ST; DO_ST;           // 1R + 2W
#elif OP == 4
        DO_LD; DO_ST; DO_ST; DO_ST;    // 1R + 3W
#elif OP == 5
        DO_LD; DO_LD; DO_ST;           // 2R + 1W
#elif OP == 6
        DO_LD; DO_LD; DO_LD; DO_ST;    // 3R + 1W
#elif OP == 7
        DO_ST; DO_ST; DO_ST; DO_ST;    // 4W
#elif OP == 8
        DO_LD; DO_LD; DO_LD; DO_LD;    // 4R
#endif
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;");
    asm volatile("tcgen05.wait::st.sync.aligned;");
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem_addr));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
    if (threadIdx.x == 0) ((unsigned long long*)C)[0] = t1 - t0;
}
