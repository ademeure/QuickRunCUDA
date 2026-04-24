// Force FULL unroll by making ITERS a compile-time constant.
// This is the "worst case" for I-cache footprint that the original §21 test
// may have used. With pragma unroll inside a runtime loop, NVCC limits unroll
// to the pragma N. But with a compile-time-known bound and #pragma unroll
// (full), it will fully unroll.
//
// Build: -H "#define MMA_KIND 1 -DSTATIC_ITERS=N"
//
// Hypothesis: catalog cliff was only seen because FULL unroll → N MMAs of
// SASS code → I-cache thrash. With UNROLL 1, body is ~96 bytes → no thrash.

#ifndef MMA_KIND
#define MMA_KIND 1
#endif
#ifndef STATIC_ITERS
#define STATIC_ITERS 100000
#endif
#ifndef UNROLL_MODE
#define UNROLL_MODE 1   // 1=full unroll, 0=unroll 1
#endif

#if MMA_KIND == 0
  #define MMA_M 64
  #define MMA_N 8
  #define TMEM_COLS 128
#elif MMA_KIND == 1
  #define MMA_M 128
  #define MMA_N 128
  #define TMEM_COLS 512
#elif MMA_KIND == 2
  #define MMA_M 128
  #define MMA_N 256
  #define TMEM_COLS 512
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int u0, int u1, int u2) {
    __shared__ __align__(1024) unsigned smem_A[3072];
    __shared__ __align__(1024) unsigned smem_B[3072];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;

    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < 3072; idx += 32) {
            smem_A[idx] = 0xDEADBEEFu ^ (idx * 0xCAFEBABEu);
            smem_B[idx] = 0x12345678u ^ (idx * 0x13579BDFu);
        }
    }
    if (threadIdx.x == 0) {
        tmem_slot = 0xFFFFFFFFu;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)));
    }
    __syncthreads();

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)), "n"(TMEM_COLS) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    __syncthreads();
    unsigned tmem_addr = tmem_slot;

    unsigned idesc = (4U << 7) | (4U << 10)
                   | (((unsigned)MMA_N >> 3) << 17)
                   | (((unsigned)MMA_M >> 4) << 24);

    auto desc_encode = [](unsigned long long x) -> unsigned long long {
        return (x & 0x3FFFFULL) >> 4;
    };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16;
    unsigned long long SBO = 2 * (unsigned long long)MMA_M;
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);

    unsigned long long t0 = 0, t1 = 0;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (threadIdx.x == 0) {
        unsigned scaleC = 0;

#if UNROLL_MODE == 1
        #pragma unroll
#else
        #pragma unroll 1
#endif
        for (int i = 0; i < STATIC_ITERS; i++) {
            asm volatile(
                "{\n\t .reg .pred P;\n\t setp.ne.b32 P, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, P;\n\t}"
                :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(scaleC)
                : "memory");
            scaleC = 1;
        }

        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");

        unsigned phase = 0;
        asm volatile(
            "{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase));

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();

    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr), "n"(TMEM_COLS));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long total_cy = t1 - t0;
        double cy_per_mma = (double)total_cy / (double)STATIC_ITERS;
        ((unsigned long long*)C)[0] = total_cy;
        printf("[STATIC_ITERS=%d UNROLL_MODE=%d] total_cy=%llu cy/MMA=%.3f\n",
               STATIC_ITERS, UNROLL_MODE, total_cy, cy_per_mma);
    }
}
