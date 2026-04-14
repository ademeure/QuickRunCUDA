// atom scope modifiers: .cta / .gpu / .sys / .cluster

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v0=0,v1=0,v2=0,v3=0,v4=0,v5=0,v6=0,v7=0;
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long base = (unsigned long long)A + tid * 32;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#define G(k) (base + (k) * 4ULL)
#if OP == 0  // default scope .gpu
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v0) : "l"(G(0)), "r"(1u));
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v1) : "l"(G(1)), "r"(1u));
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v2) : "l"(G(2)), "r"(1u));
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v3) : "l"(G(3)), "r"(1u));
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v4) : "l"(G(4)), "r"(1u));
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v5) : "l"(G(5)), "r"(1u));
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v6) : "l"(G(6)), "r"(1u));
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v7) : "l"(G(7)), "r"(1u));
#elif OP == 1 // atom.global.cta.add (cooperative thread array scope)
            asm volatile("atom.cta.global.add.u32 %0, [%1], %2;" : "=r"(v0) : "l"(G(0)), "r"(1u));
            asm volatile("atom.cta.global.add.u32 %0, [%1], %2;" : "=r"(v1) : "l"(G(1)), "r"(1u));
            asm volatile("atom.cta.global.add.u32 %0, [%1], %2;" : "=r"(v2) : "l"(G(2)), "r"(1u));
            asm volatile("atom.cta.global.add.u32 %0, [%1], %2;" : "=r"(v3) : "l"(G(3)), "r"(1u));
            asm volatile("atom.cta.global.add.u32 %0, [%1], %2;" : "=r"(v4) : "l"(G(4)), "r"(1u));
            asm volatile("atom.cta.global.add.u32 %0, [%1], %2;" : "=r"(v5) : "l"(G(5)), "r"(1u));
            asm volatile("atom.cta.global.add.u32 %0, [%1], %2;" : "=r"(v6) : "l"(G(6)), "r"(1u));
            asm volatile("atom.cta.global.add.u32 %0, [%1], %2;" : "=r"(v7) : "l"(G(7)), "r"(1u));
#elif OP == 2 // atom.sys.global.add (system scope — includes CPU coherence)
            asm volatile("atom.sys.global.add.u32 %0, [%1], %2;" : "=r"(v0) : "l"(G(0)), "r"(1u));
            asm volatile("atom.sys.global.add.u32 %0, [%1], %2;" : "=r"(v1) : "l"(G(1)), "r"(1u));
            asm volatile("atom.sys.global.add.u32 %0, [%1], %2;" : "=r"(v2) : "l"(G(2)), "r"(1u));
            asm volatile("atom.sys.global.add.u32 %0, [%1], %2;" : "=r"(v3) : "l"(G(3)), "r"(1u));
            asm volatile("atom.sys.global.add.u32 %0, [%1], %2;" : "=r"(v4) : "l"(G(4)), "r"(1u));
            asm volatile("atom.sys.global.add.u32 %0, [%1], %2;" : "=r"(v5) : "l"(G(5)), "r"(1u));
            asm volatile("atom.sys.global.add.u32 %0, [%1], %2;" : "=r"(v6) : "l"(G(6)), "r"(1u));
            asm volatile("atom.sys.global.add.u32 %0, [%1], %2;" : "=r"(v7) : "l"(G(7)), "r"(1u));
#elif OP == 3 // atom.relaxed.gpu.global.add (relaxed memory order, weaker than .strong)
            asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v0) : "l"(G(0)), "r"(1u));
            asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v1) : "l"(G(1)), "r"(1u));
            asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v2) : "l"(G(2)), "r"(1u));
            asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v3) : "l"(G(3)), "r"(1u));
            asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v4) : "l"(G(4)), "r"(1u));
            asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v5) : "l"(G(5)), "r"(1u));
            asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v6) : "l"(G(6)), "r"(1u));
            asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v7) : "l"(G(7)), "r"(1u));
#elif OP == 4 // atom.acq_rel.gpu.global.add (acquire-release ordering)
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v0) : "l"(G(0)), "r"(1u));
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v1) : "l"(G(1)), "r"(1u));
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v2) : "l"(G(2)), "r"(1u));
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v3) : "l"(G(3)), "r"(1u));
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v4) : "l"(G(4)), "r"(1u));
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v5) : "l"(G(5)), "r"(1u));
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v6) : "l"(G(6)), "r"(1u));
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v7) : "l"(G(7)), "r"(1u));
#endif
        }
    }
    unsigned int acc = v0^v1^v2^v3^v4^v5^v6^v7;
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
