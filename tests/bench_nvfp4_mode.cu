#ifndef MMA_K
#define MMA_K 96
#endif
#ifndef MMA_M
#define MMA_M 256
#endif
#ifndef MMA_N
#define MMA_N 256
#endif
extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void kernel(float* A, float* B, float* C, int iters, int mode, int u2) {
    unsigned laneid; asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
    if (laneid != 0) return;
    extern __shared__ unsigned smem[];
    int a_size = MMA_M * MMA_K / 8, b_size = MMA_K * MMA_N / 8;
    unsigned* smem_A = smem;
    unsigned* smem_B = smem + a_size;
    unsigned long long* mbar_p = (unsigned long long*)(smem_B + b_size);
    unsigned* tmem_p = (unsigned*)(mbar_p + 1);
    for (int i = 0; i < a_size; i++) {
        int k = i / (MMA_M/8);
        int mpack = i % (MMA_M/8);
        unsigned seed;
        if (mode == 13) seed = k;       // A uniform-M: varies along K only
        else if (mode == 14) seed = mpack;  // A uniform-K: varies along M only
        else seed = i;
        unsigned r = (seed + blockIdx.x * 1024u) * 0x9E3779B1u; r ^= r >> 16; r *= 0x85EBCA6Bu;
        if (mode == 2 || mode == 3) r &= ~0x88888888u;
        if (mode == 5 || mode == 7) r = 0u;
        if (mode == 10) r = 0x22222222u;
        smem_A[i] = r;
    }
    for (int i = 0; i < b_size; i++) {
        int k = i / (MMA_N/8);
        int npack = i % (MMA_N/8);
        unsigned seed;
        if (mode == 11) seed = npack;
        else if (mode == 12) seed = k;
        else if (mode == 15) seed = (k / 16) * (MMA_N/8) + npack;  // B uniform per K-chunk-of-16
        else if (mode == 16) seed = (k / 32) * (MMA_N/8) + npack;  // B uniform per K-chunk-of-32
        else if (mode == 17) seed = (k / 8) * (MMA_N/8) + npack;   // B uniform per K-chunk-of-8
        else if (mode == 18) seed = (k / 4) * (MMA_N/8) + npack;   // K-chunk-of-4
        else if (mode == 19) seed = (k / 2) * (MMA_N/8) + npack;
        else if (mode == 20) seed = (k / 48) * (MMA_N/8) + npack;  // K-chunk-of-48 (only 1 transition!)
        else seed = i;
        unsigned r = (seed + blockIdx.x * 1024u + 0xC0FFEE00u) * 0x9E3779B1u; r ^= r >> 16; r *= 0x85EBCA6Bu;
        // Specialized B value sets, override after random
        if (mode == 21) {
            // B uniformly chosen from {+1.0, +1.5} - tiny magnitude variation, no sign
            r = ((r & 0x11111111u) ? 0x33333333u : 0x22222222u);
        } else if (mode == 22) {
            // B uniformly chosen from {+0.5, +1.0, +1.5, +2.0} - 4 close positives, no sign
            unsigned b = 0;
            for (int p = 0; p < 8; p++) { b |= (((r >> (p*4)) & 3) + 1) << (p*4); }
            r = b;
        } else if (mode == 23) {
            // B uniformly chosen from {-6.0, +6.0} - extreme bipolar
            r = ((r & 0x11111111u) ? 0xFFFFFFFFu : 0x77777777u);
        } else if (mode == 24) {
            r = ((r & 0x11111111u) ? 0xAAAAAAAAu : 0x22222222u);
        } else if (mode == 25) {
            // 4 distinct DWORD patterns (4 nibble values 0,1,2,3 splatted)
            unsigned t = r & 3;
            r = (t * 0x11111111u);  // 0x00000000, 0x11111111, 0x22222222, 0x33333333
        } else if (mode == 26) {
            // 8 distinct DWORD patterns
            unsigned t = r & 7;
            r = (t * 0x11111111u);
        } else if (mode == 27) {
            unsigned t = r & 0xF;
            r = (t * 0x11111111u);
        }
        // 2-dword Hamming-distance series (mode 30 + N: distance = N)
        // mode 30 = both dwords identical (0 bit diff)
        // mode 31 = 1-bit diff, mode 32 = 2-bit, ... mode 36 = 16-bit, mode 37 = 32-bit
        else if (mode == 30) r = 0x00000000u;
        else if (mode == 31) r = ((r & 1) ? 0x00000001u : 0x00000000u);
        else if (mode == 32) r = ((r & 1) ? 0x00000003u : 0x00000000u);
        else if (mode == 33) r = ((r & 1) ? 0x0000000Fu : 0x00000000u);
        else if (mode == 34) r = ((r & 1) ? 0x000000FFu : 0x00000000u);
        else if (mode == 35) r = ((r & 1) ? 0x0000FFFFu : 0x00000000u);
        else if (mode == 36) r = ((r & 1) ? 0x00FFFFFFu : 0x00000000u);
        else if (mode == 37) r = ((r & 1) ? 0xFFFFFFFFu : 0x00000000u);
        // Sequential dwords: many patterns but only 1-bit Hamming between consecutive
        // mode 40: dwords cycle 0, 1, 3, 7, 15, ... (each adds 1 bit)
        else if (mode == 40) {
            // Gray-code-like: each dword differs from prior by 1 bit
            // For each `seed` (which is `i` for mode 40), use seed mod 32 = bits set
            // Construct dword with `seed mod 32` low bits set
            unsigned t = seed % 32;
            r = (t == 0) ? 0u : (0xFFFFFFFFu >> (32 - t));
        }
        // mode 41: only 2-bit Hamming via XOR of seed
        else if (mode == 41) {
            // Many distinct dwords each differing from neighbor by 2 bits
            unsigned base = (seed * 0xCAFEBABE) & 0xFFFFFFFFu;
            // Force base to have exactly 16 bits set (popcount-controlled)
            unsigned popcount_target = 16;
            // Quick approximation: just XOR with seed pattern
            r = base ^ ((seed & 1) ? 0x3 : 0);  // 2-bit toggle between consecutive
        }
        if (mode == 1 || mode == 3) r &= ~0x88888888u;
        if (mode == 6 || mode == 7) r = 0u;
        if (mode == 9) r = 0xAAAAAAAAu;
        if (mode == 8) r = 0x22222222u;
        smem_B[i] = r;
    }
    *tmem_p = 0xFFFFFFFFu;
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(mbar_p)));
    asm volatile("barrier.cluster.arrive.aligned;"); asm volatile("barrier.cluster.wait.aligned;");
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;"
        :: "r"((unsigned)__cvta_generic_to_shared(tmem_p)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    asm volatile("barrier.cluster.arrive.aligned;"); asm volatile("barrier.cluster.wait.aligned;");
    unsigned tmem_addr = *tmem_p;
    unsigned tsfa_addr = tmem_addr + 128, tsfb_addr = tmem_addr + 256;
    { unsigned sf_seed = (mode == 4) ? (blockIdx.x * 13u + threadIdx.x) * 0x9E3779B1u : 0;
      unsigned one = (mode == 4) ? sf_seed : 0x38383838u;
      for (int ch=0; ch<4; ch++) {
        unsigned addr = tmem_addr + ch*128;
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
          :: "r"(addr),"r"(one),"r"(one),"r"(one),"r"(one));
      }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;");
#if MMA_K == 96
    unsigned idesc = (5U<<7)|(5U<<10)|(((unsigned)MMA_N>>3)<<17)|(((unsigned)MMA_M>>4)<<24)|(1U<<31);
#else
    unsigned idesc = (5U<<7)|(5U<<10)|(((unsigned)MMA_N>>3)<<17)|(((unsigned)MMA_M>>4)<<24);
#endif
    auto denc = [](unsigned long long x) -> unsigned long long { return (x & 0x3FFFFULL) >> 4; };
    unsigned a_sa = (unsigned)__cvta_generic_to_shared(smem_A), b_sa = (unsigned)__cvta_generic_to_shared(smem_B);
    unsigned long long LBO = 16, SBO = 2 * (unsigned long long)(MMA_M / 2);
    unsigned long long a_desc = denc(a_sa) | (denc(LBO) << 16) | (denc(SBO) << 32);
    unsigned long long b_desc = denc(b_sa) | (denc(LBO) << 16) | (denc(SBO) << 32);
    if ((blockIdx.x & 1) == 0) {
        unsigned scaleC = 0;
        for (int i = 0; i < iters; i++) {
            asm volatile("{\n\t .reg .pred P;\n\t setp.ne.b32 P, %4, 0;\n\t"
                "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], P;\n\t}"
                :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                   "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr) : "memory");
            scaleC = 1;
        }
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];"
            :: "r"((unsigned)__cvta_generic_to_shared(mbar_p)) : "memory");
        unsigned ph = 0;
        asm volatile("{\n\t .reg .pred P;\n\t WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t @P bra DONE;\n\t bra WAIT;\n\t DONE:\n\t}"
            :: "r"((unsigned)__cvta_generic_to_shared(mbar_p)), "r"(ph));
    }
    asm volatile("barrier.cluster.arrive.aligned;"); asm volatile("barrier.cluster.wait.aligned;");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
}
