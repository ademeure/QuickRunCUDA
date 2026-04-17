// Bit-manipulation ops audit: POPC, BREV, CLZ, BFIND, BFE, BFI, PRMT.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ITERS
#define ITERS 4096
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned v = (unsigned)seed + threadIdx.x;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if OP == 0  // POPC
        v = __popc(v);
        v += i;
#elif OP == 1  // BREV
        v = __brev(v);
        v += i;
#elif OP == 2  // CLZ (count leading zeros)
        v = __clz(v);
        v += i;
#elif OP == 3  // FFS (find first set)
        v = __ffs(v);
        v += i;
#elif OP == 4  // BFE (bit field extract)
        unsigned r;
        asm volatile("bfe.u32 %0, %1, 8, 16;" : "=r"(r) : "r"(v));
        v = r + i;
#elif OP == 5  // BFI (bit field insert)
        unsigned r;
        asm volatile("bfi.b32 %0, %1, %2, 8, 16;" : "=r"(r) : "r"(v), "r"(i));
        v = r;
#elif OP == 6  // PRMT (byte permute)
        unsigned r;
        asm volatile("prmt.b32 %0, %1, %2, 0x5432;" : "=r"(r) : "r"(v), "r"(i));
        v = r;
#elif OP == 7  // ISETP + SEL
        v = (v > (unsigned)i) ? v+1 : v-1;
#elif OP == 8  // SHF (funnel shift)
        unsigned r;
        asm volatile("shf.l.wrap.b32 %0, %1, %2, 7;" : "=r"(r) : "r"(v), "r"(i));
        v = r;
#endif
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = v;
    }
}
