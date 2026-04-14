// Extended catalog of less-common PTX ops.

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
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
    unsigned int v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) v[k] = 0xDEADBEEFu ^ (threadIdx.x * 131 + k * 17);
    if (threadIdx.x < 1024) smem[threadIdx.x] = threadIdx.x;
    __syncthreads();

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned int x = v[k];
                unsigned int nxt = v[(k+1) & (N_CHAINS-1)];

#if OP == 0   // fma.rn.f32 with immediate (expects FFMA32I)
                {float f = __int_as_float(x);
                 asm volatile("fma.rn.f32 %0, %0, 0f3F800001, 0f3F7FFFBF;" : "+f"(f));
                 x = __float_as_int(f);}
#elif OP == 1 // add.u32 with immediate (IADD32I)
                asm volatile("add.u32 %0, %0, 0x12345;" : "+r"(x));
#elif OP == 2 // mul.lo.u32 immediate (IMUL32I)
                asm volatile("mul.lo.u32 %0, %0, 0x12345;" : "+r"(x));
#elif OP == 3 // xor.b32 immediate (LOP32I)
                asm volatile("xor.b32 %0, %0, 0xDEADBEEF;" : "+r"(x));
#elif OP == 4 // ISCADD (shift + add — no direct PTX, use `mad.lo` with power of 2)
                asm volatile("mad.lo.u32 %0, %0, 4, %1;" : "+r"(x) : "r"(nxt));  // emits IMAD, not ISCADD
#elif OP == 5 // FSWZADD (swizzle add — no PTX, only CUDA __builtin)
                asm volatile("fswzadd.rn.f32 %0, %1, %1, 0x88;" : "=f"(*(float*)&x) : "f"(__int_as_float(nxt)));
#elif OP == 6 // FMNMX3 via PTX (min3.f32 — may not exist)
                {float f=__int_as_float(x),g=__int_as_float(nxt);
                 asm volatile("min.f32 %0, %0, %1; min.f32 %0, %0, %2;" : "+f"(f) : "f"(g), "f"(0.5f));
                 x=__float_as_int(f);}
#elif OP == 7 // LDG with .ca cache hint
                asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"((unsigned long long)(A + (threadIdx.x + k) % 4096)));
#elif OP == 8 // LDG with .cg cache hint
                asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"((unsigned long long)(A + (threadIdx.x + k) % 4096)));
#elif OP == 9 // LDG with .lu cache hint (last use)
                asm volatile("ld.global.lu.u32 %0, [%1];" : "=r"(x) : "l"((unsigned long long)(A + (threadIdx.x + k) % 4096)));
#elif OP == 10 // STG with .wb (write back)
                asm volatile("st.global.wb.u32 [%0], %1;" :: "l"((unsigned long long)(B + threadIdx.x + k * 1024)), "r"(x));
#elif OP == 11 // STG with .cs (streaming)
                asm volatile("st.global.cs.u32 [%0], %1;" :: "l"((unsigned long long)(B + threadIdx.x + k * 1024)), "r"(x));
#elif OP == 12 // atom.shared.min
                asm volatile("atom.shared.min.u32 %0, [%1], %2;" : "=r"(x) : "r"((unsigned)__cvta_generic_to_shared(&smem[(threadIdx.x + k) & 0x3FF])), "r"(nxt));
#elif OP == 13 // atom.shared.cas
                asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(x) : "r"((unsigned)__cvta_generic_to_shared(&smem[(threadIdx.x + k) & 0x3FF])), "r"(x), "r"(nxt));
#elif OP == 14 // atom.shared.exch
                asm volatile("atom.shared.exch.b32 %0, [%1], %2;" : "=r"(x) : "r"((unsigned)__cvta_generic_to_shared(&smem[(threadIdx.x + k) & 0x3FF])), "r"(nxt));
#elif OP == 15 // cvt.rz.f32.f32 (round-toward-zero self)
                {float f = __int_as_float(x);
                 asm volatile("cvt.rz.ftz.f32.f32 %0, %1;" : "=f"(f) : "f"(f));
                 x = __float_as_int(f);}
#elif OP == 16 // testp.normal.f32
                {float f = __int_as_float(x);
                 asm volatile("{.reg .pred p; testp.normal.f32 p, %1; selp.u32 %0, 1, 2, p;}" : "=r"(x) : "f"(f));}
#elif OP == 17 // bfind.u32
                asm volatile("bfind.u32 %0, %0;" : "+r"(x));
#elif OP == 18 // bfind.shiftamt.u32
                asm volatile("bfind.shiftamt.u32 %0, %0;" : "+r"(x));
#elif OP == 19 // sad.u32
                asm volatile("sad.u32 %0, %0, %1, %2;" : "+r"(x) : "r"(nxt), "r"(x));
#elif OP == 20 // NANOSLEEP (brief)
                asm volatile("nanosleep.u32 %0;" :: "n"(8));
#elif OP == 21 // elect.sync
                asm volatile("{.reg .pred p; elect.sync %0|p, 0xFFFFFFFF; }" : "=r"(x));
#elif OP == 22 // cp.async (CUDA 11+) — 4-byte from global to shared
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::
                    "r"((unsigned)__cvta_generic_to_shared(&smem[(threadIdx.x + k) & 0x3FF])),
                    "l"((unsigned long long)(A + (threadIdx.x + k) % 4096)));
#elif OP == 23 // cp.async.bulk group commit/wait pairs — simplified
                asm volatile("cp.async.commit_group;");
#elif OP == 24 // prefetch.global
                asm volatile("prefetch.global.L1 [%0];" :: "l"((unsigned long long)(A + (threadIdx.x + k) % 4096)));
#elif OP == 25 // prefetch.global.L2
                asm volatile("prefetch.global.L2 [%0];" :: "l"((unsigned long long)(A + (threadIdx.x + k) % 4096)));
#elif OP == 26 // vabsdiff.s32 (signed version)
                asm volatile("vabsdiff.s32.s32.s32 %0, %0, %1, %0;" : "+r"(x) : "r"(nxt));
#elif OP == 27 // LEA-like: mad.lo + shift — test if compiler emits LEA
                asm volatile("mad.wide.u32 %0, %0, 8, %1;" : "=l"(*(unsigned long long*)&x) : "r"(nxt));
#elif OP == 28 // predicate logic plop3: setp, and/or/xor pred
                asm volatile("{.reg .pred p,q; setp.eq.u32 p, %0, 0; setp.eq.u32 q, %0, 1; and.pred p, p, q; selp.u32 %0, 1, 2, p; }" : "+r"(x));
#elif OP == 29 // S2R SR_CTAID.X
                asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(x));
#elif OP == 30 // S2R %nctaid.x
                asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(x));
#elif OP == 31 // fma.rn.ftz.f32 (flush to zero)
                {float f = __int_as_float(x);
                 asm volatile("fma.rn.ftz.f32 %0, %0, %1, %2;" : "+f"(f) : "f"(1.000001f), "f"(0.9999f));
                 x = __float_as_int(f);}
#endif
                v[k] = x;
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
