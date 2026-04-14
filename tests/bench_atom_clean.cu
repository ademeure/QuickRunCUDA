// Atomics with BANK-CLEAN addresses: stride 4B → lane t hits bank t.
// Also each chain uses different "page" so no repeat-address serialisation.

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

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v0=0,v1=1,v2=2,v3=3,v4=4,v5=5,v6=6,v7=7;
    // Lane t gets bank t: base[t] = t * 4, chain k in separate page
    // page[k] = k * BLOCK_SIZE * 4 bytes
    unsigned int tid = threadIdx.x;
    if (tid < BLOCK_SIZE * 8) smem[tid] = tid;
    __syncthreads();

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#define ADR(k) ((unsigned)__cvta_generic_to_shared(&smem[tid + (k) * BLOCK_SIZE]))
#if OP == 0
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v0) : "r"(ADR(0)), "r"(1u));
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v1) : "r"(ADR(1)), "r"(1u));
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v2) : "r"(ADR(2)), "r"(1u));
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v3) : "r"(ADR(3)), "r"(1u));
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v4) : "r"(ADR(4)), "r"(1u));
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v5) : "r"(ADR(5)), "r"(1u));
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v6) : "r"(ADR(6)), "r"(1u));
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(v7) : "r"(ADR(7)), "r"(1u));
#elif OP == 1
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v0) : "r"(ADR(0)), "r"(tid), "r"(tid+1));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v1) : "r"(ADR(1)), "r"(tid), "r"(tid+1));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v2) : "r"(ADR(2)), "r"(tid), "r"(tid+1));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v3) : "r"(ADR(3)), "r"(tid), "r"(tid+1));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v4) : "r"(ADR(4)), "r"(tid), "r"(tid+1));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v5) : "r"(ADR(5)), "r"(tid), "r"(tid+1));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v6) : "r"(ADR(6)), "r"(tid), "r"(tid+1));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(v7) : "r"(ADR(7)), "r"(tid), "r"(tid+1));
#elif OP == 2  // LDS (reference — should saturate lsu)
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v0) : "r"(ADR(0)));
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v1) : "r"(ADR(1)));
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v2) : "r"(ADR(2)));
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v3) : "r"(ADR(3)));
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v4) : "r"(ADR(4)));
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v5) : "r"(ADR(5)));
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v6) : "r"(ADR(6)));
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v7) : "r"(ADR(7)));
#endif
        }
    }
    unsigned int acc = v0^v1^v2^v3^v4^v5^v6^v7;
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
