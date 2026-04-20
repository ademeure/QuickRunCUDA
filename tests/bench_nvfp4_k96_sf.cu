#define MMA_M 256
#define MMA_N 256
#define MMA_K 96

extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void kernel(float* A, float* B, float* C, int iters, int sf_mode, int u2) {
    // sf_mode:
    //   0 = SF = 0x38 (UE4M3 = 1.0) — baseline
    //   1 = SF = 0x00 (zero scale → result is 0)
    //   2 = SF = 0xFF (max scale)
    //   3 = SF = random byte
    //   4 = SF = alternating 0x38 / 0x00 (every 16 elements)
    //   5 = SF = uniform in {0x37, 0x38, 0x39} (3 close values)
    __shared__ __align__(1024) unsigned smem_A[3072];
    __shared__ __align__(1024) unsigned smem_B[3072];
    __shared__ __align__(8)    unsigned long long mbar;
    __shared__ __align__(4)    unsigned tmem_slot;
    int smem_size = MMA_K * MMA_N / 8;
    if (threadIdx.x < 32) {
        for (int idx = threadIdx.x; idx < smem_size; idx += 32) {
            unsigned r = (idx + blockIdx.x * 1024u) * 0x9E3779B1u;
            r ^= r >> 16; r *= 0x85EBCA6Bu;
            r ^= r >> 13; r *= 0xC2B2AE35u;
            r ^= r >> 16;
            smem_A[idx] = r;       // A random
            smem_B[idx] = r ^ 0xCAFEBABEu;   // B random (different)
        }
    }
    if (threadIdx.x == 0) {
        tmem_slot = 0xFFFFFFFFu;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)));
    }
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(&tmem_slot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    unsigned tmem_addr = tmem_slot;
    unsigned tsfa_addr = tmem_addr + 128;
    unsigned tsfb_addr = tmem_addr + 256;
    {
        for (int chunk = 0; chunk < 4; chunk++) {
            unsigned col_base = chunk * 128 + (threadIdx.x * 4);
            unsigned addr = tmem_addr + col_base;
            unsigned pa, pb, pc, pd;
            switch (sf_mode) {
                case 0: pa = pb = pc = pd = 0x38383838u; break;
                case 1: pa = pb = pc = pd = 0x00000000u; break;
                case 2: pa = pb = pc = pd = 0xFFFFFFFFu; break;
                case 3: {
                    unsigned s = (chunk * 32u + threadIdx.x) * 0x9E3779B1u + 0xC0FFEE00u;
                    s ^= s >> 16; s *= 0x85EBCA6Bu; s ^= s >> 13;
                    pa = s + 0x12345678u; pb = s ^ 0xABCDEF01u;
                    pc = s + 0xCAFEBABEu; pd = s ^ 0xDEADBEEFu;
                    break;
                }
                case 4: pa = pc = 0x38383838u; pb = pd = 0x00000000u; break;
                case 5: {
                    // Uniform in {0x37, 0x38, 0x39}
                    unsigned s = (chunk * 32u + threadIdx.x) * 0x9E3779B1u;
                    auto pick = [](unsigned x) -> unsigned char {
                        unsigned char b[3] = {0x37, 0x38, 0x39};
                        return b[(x % 3u)];
                    };
                    pa = pick(s) | (pick(s+1) << 8) | (pick(s+2) << 16) | (pick(s+3) << 24);
                    pb = pick(s+4) | (pick(s+5) << 8) | (pick(s+6) << 16) | (pick(s+7) << 24);
                    pc = pick(s+8) | (pick(s+9) << 8) | (pick(s+10) << 16) | (pick(s+11) << 24);
                    pd = pick(s+12) | (pick(s+13) << 8) | (pick(s+14) << 16) | (pick(s+15) << 24);
                    break;
                }
                default: pa = pb = pc = pd = 0x38383838u;
            }
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};\n"
                :: "r"(addr), "r"(pa), "r"(pb), "r"(pc), "r"(pd));
        }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;");
    __syncthreads();
    unsigned idesc = (5U << 7) | (5U << 10) | (((unsigned)MMA_N >> 3) << 17)
                   | (((unsigned)MMA_M >> 4) << 24) | (1U << 31);
    auto desc_encode = [](unsigned long long x) -> unsigned long long { return (x & 0x3FFFFULL) >> 4; };
    unsigned a_smem_addr = (unsigned)__cvta_generic_to_shared(smem_A);
    unsigned b_smem_addr = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16;
    unsigned long long SBO = 2 * (unsigned long long)(MMA_M / 2);
    unsigned long long a_desc = desc_encode(a_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    unsigned long long b_desc = desc_encode(b_smem_addr) | (desc_encode(LBO) << 16) | (desc_encode(SBO) << 32);
    if ((blockIdx.x % 2) == 0 && threadIdx.x == 0) {
        unsigned scaleC = 0;
        for (int i = 0; i < iters; i++) {
            asm volatile(
                "{\n\t .reg .pred PRED;\n\t setp.ne.b32 PRED, %4, 0;\n\t"
                "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], PRED;\n\t}"
                :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                   "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr) : "memory");
            scaleC = 1;
        }
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
        unsigned phase_w = 0;
        asm volatile("{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t @P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(phase_w));
    }
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
}
