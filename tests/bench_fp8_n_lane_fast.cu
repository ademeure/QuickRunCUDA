// FP8 (E4M3) tcgen05 N-sweep with laneid early-exit
#define MMA_M 256
#define MMA_K 32
#ifndef MMA_N
#define MMA_N 256
#endif

extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void kernel(float* A, float* B, float* C, int iters, int mode, int u2) {
    unsigned laneid;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
    if (laneid != 0) return;

    extern __shared__ unsigned smem[];
    int a_size = MMA_M * MMA_K / 4;      // FP8: 4 per dword
    int b_size = MMA_K * MMA_N / 4;
    unsigned* smem_A = smem;
    unsigned* smem_B = smem + a_size;
    unsigned long long* mbar_p = (unsigned long long*)(smem_B + b_size);
    unsigned* tmem_p = (unsigned*)(mbar_p + 1);

    for (int i = 0; i < a_size; i++) {
        unsigned r = (i + blockIdx.x * 1024u) * 0x9E3779B1u;
        r ^= r >> 16; r *= 0x85EBCA6Bu; r ^= r >> 13; r *= 0xC2B2AE35u; r ^= r >> 16;
        smem_A[i] = r;
    }
    for (int i = 0; i < b_size; i++) {
        unsigned r = (i + blockIdx.x * 1024u + 0xC0FFEE00u) * 0x9E3779B1u;
        r ^= r >> 16; r *= 0x85EBCA6Bu; r ^= r >> 13; r *= 0xC2B2AE35u; r ^= r >> 16;
        smem_B[i] = r;
    }

    *tmem_p = 0xFFFFFFFFu;
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
        :: "r"((unsigned)__cvta_generic_to_shared(mbar_p)));
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(tmem_p)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    unsigned tmem_addr = *tmem_p;

    // FP8 E4M3: a_format=0, b_format=0
    unsigned idesc = (1U << 4) | (0U << 7) | (0U << 10)
                   | (((unsigned)MMA_N >> 3) << 17)
                   | (((unsigned)MMA_M >> 4) << 24);
    auto denc = [](unsigned long long x) -> unsigned long long { return (x & 0x3FFFFULL) >> 4; };
    unsigned a_sa = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_sa = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16;
    unsigned long long SBO = 2 * (unsigned long long)(MMA_M / 2);
    unsigned long long a_desc = denc(a_sa) | (denc(LBO) << 16) | (denc(SBO) << 32);
    unsigned long long b_desc = denc(b_sa) | (denc(LBO) << 16) | (denc(SBO) << 32);
    unsigned dl0=0, dl1=0, dl2=0, dl3=0, dl4=0, dl5=0, dl6=0, dl7=0;

    if ((blockIdx.x & 1) == 0) {
        unsigned long long t0=0, t1=0;
        if (blockIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned enable_d = 0;
        for (int i = 0; i < iters; i++) {
            asm volatile(
                "{\n\t .reg .pred P;\n\t setp.ne.b32 P, %12, 0;\n\t"
                "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, "
                "{%4,%5,%6,%7,%8,%9,%10,%11}, P;\n\t}"
                :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                   "r"(dl0),"r"(dl1),"r"(dl2),"r"(dl3),"r"(dl4),"r"(dl5),"r"(dl6),"r"(dl7),
                   "r"(enable_d) : "memory");
            enable_d = 1;
        }
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(mbar_p)) : "memory");
        unsigned ph = 0;
        asm volatile("{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t @P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(mbar_p)), "r"(ph));
        if (blockIdx.x == 0) {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
            printf("FP8 M=%d N=%d K=%d cy/MMA=%.2f\n", MMA_M, MMA_N, MMA_K, (double)(t1-t0)/iters);
        }
    }
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
}
