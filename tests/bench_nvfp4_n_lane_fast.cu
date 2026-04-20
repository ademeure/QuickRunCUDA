// NVFP4 tcgen05 N-sweep parameterized K + CTA_GROUP
#ifndef MMA_M
#define MMA_M 256
#endif
#ifndef MMA_K
#define MMA_K 96
#endif
#ifndef MMA_N
#define MMA_N 256
#endif
#ifndef CTA_GROUP
#define CTA_GROUP 2
#endif

#if CTA_GROUP == 2
extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
#else
extern "C" __global__ __launch_bounds__(32, 1)
#endif
void kernel(float* A, float* B, float* C, int iters, int mode, int u2) {
    unsigned laneid;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
    if (laneid != 0) return;

    extern __shared__ unsigned smem[];
    int a_size = MMA_M * MMA_K / 8;
    int b_size = MMA_K * MMA_N / 8;
    unsigned* smem_A = smem;
    unsigned* smem_B = smem + a_size;
    unsigned long long* mbar_p = (unsigned long long*)(smem_B + b_size);
    unsigned* tmem_p = (unsigned*)(mbar_p + 1);

    for (int i = 0; i < a_size; i++) {
        unsigned r = (i + blockIdx.x * 1024u) * 0x9E3779B1u;
        r ^= r >> 16; r *= 0x85EBCA6Bu;
        smem_A[i] = r;
    }
    for (int i = 0; i < b_size; i++) {
        unsigned r = (i + blockIdx.x * 1024u + 0xC0FFEE00u) * 0x9E3779B1u;
        r ^= r >> 16; r *= 0x85EBCA6Bu;
        smem_B[i] = r;
    }

    *tmem_p = 0xFFFFFFFFu;
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
        :: "r"((unsigned)__cvta_generic_to_shared(mbar_p)));

#if CTA_GROUP == 2
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(tmem_p)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
#else
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(tmem_p)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
#endif
    unsigned tmem_addr = *tmem_p;
    unsigned tsfa_addr = tmem_addr + 128;
    unsigned tsfb_addr = tmem_addr + 256;

    { unsigned one = 0x38383838u;  // UE4M3 = 1.0
      for (int ch=0; ch<4; ch++) {
        unsigned addr = tmem_addr + ch*128;
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
          :: "r"(addr),"r"(one),"r"(one),"r"(one),"r"(one));
      }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;");

    // K=96 ULTRA needs k_size_=1 (bit 31). K=64 uses default (bit 31 = 0).
#if MMA_K == 96
    unsigned idesc = (5U<<7)|(5U<<10)|(((unsigned)MMA_N>>3)<<17)|(((unsigned)MMA_M>>4)<<24)|(1U<<31);
#else
    unsigned idesc = (5U<<7)|(5U<<10)|(((unsigned)MMA_N>>3)<<17)|(((unsigned)MMA_M>>4)<<24);
#endif
    auto denc = [](unsigned long long x) -> unsigned long long { return (x & 0x3FFFFULL) >> 4; };
    unsigned a_sa = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_sa = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16;
    unsigned long long SBO = 2 * (unsigned long long)(MMA_M / 2);
    unsigned long long a_desc = denc(a_sa) | (denc(LBO) << 16) | (denc(SBO) << 32);
    unsigned long long b_desc = denc(b_sa) | (denc(LBO) << 16) | (denc(SBO) << 32);

#if CTA_GROUP == 2
    if ((blockIdx.x & 1) == 0) {
#else
    if (true) {
#endif
        unsigned long long t0=0, t1=0;
        if (blockIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned scaleC = 0;
        for (int i = 0; i < iters; i++) {
            asm volatile(
                "{\n\t .reg .pred P;\n\t setp.ne.b32 P, %4, 0;\n\t"
                "tcgen05.mma.cta_group::"
#if CTA_GROUP == 2
                "2"
#else
                "1"
#endif
                ".kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], P;\n\t}"
                :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                   "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr) : "memory");
            scaleC = 1;
        }
        asm volatile("tcgen05.commit.cta_group::"
#if CTA_GROUP == 2
            "2"
#else
            "1"
#endif
            ".mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(mbar_p)) : "memory");
        unsigned ph = 0;
        asm volatile("{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t @P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(mbar_p)), "r"(ph));
        if (blockIdx.x == 0) {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
            printf("NVFP4 M=%d N=%d K=%d cta=%d cy/MMA=%.2f\n",
                   MMA_M, MMA_N, MMA_K, CTA_GROUP, (double)(t1-t0)/iters);
        }
    }
#if CTA_GROUP == 2
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
#else
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
#endif
}
