// More SASS coverage: cvt rounding modes, F2F variants, HSET/HSETP,
// carry-chain ops, STSM, uniform-datapath forcing, cluster barriers.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
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
    unsigned int v0=threadIdx.x,v1=threadIdx.x*3,v2=threadIdx.x*5,v3=threadIdx.x*7;
    unsigned int v4=threadIdx.x*11,v5=threadIdx.x*13,v6=threadIdx.x*17,v7=threadIdx.x*19;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned int x = v0;
#if OP == 0   // cvt.rn.f32.s32 (round-nearest)
            { float f; asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v0)); v0=__float_as_int(f);
              asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v1)); v1=__float_as_int(f);
              asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v2)); v2=__float_as_int(f);
              asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v3)); v3=__float_as_int(f);
              asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v4)); v4=__float_as_int(f);
              asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v5)); v5=__float_as_int(f);
              asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v6)); v6=__float_as_int(f);
              asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v7)); v7=__float_as_int(f); }
#elif OP == 1  // cvt.rz.f32.s32 (round-toward-zero)
            { float f; asm volatile("cvt.rz.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v0)); v0=__float_as_int(f);
              asm volatile("cvt.rz.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v1)); v1=__float_as_int(f);
              asm volatile("cvt.rz.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v2)); v2=__float_as_int(f);
              asm volatile("cvt.rz.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v3)); v3=__float_as_int(f);
              asm volatile("cvt.rz.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v4)); v4=__float_as_int(f);
              asm volatile("cvt.rz.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v5)); v5=__float_as_int(f);
              asm volatile("cvt.rz.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v6)); v6=__float_as_int(f);
              asm volatile("cvt.rz.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v7)); v7=__float_as_int(f); }
#elif OP == 2  // cvt.rm.f32.s32 (round-down)
            { float f; asm volatile("cvt.rm.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v0)); v0=__float_as_int(f);
              asm volatile("cvt.rm.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v1)); v1=__float_as_int(f);
              asm volatile("cvt.rm.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v2)); v2=__float_as_int(f);
              asm volatile("cvt.rm.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v3)); v3=__float_as_int(f);
              asm volatile("cvt.rm.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v4)); v4=__float_as_int(f);
              asm volatile("cvt.rm.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v5)); v5=__float_as_int(f);
              asm volatile("cvt.rm.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v6)); v6=__float_as_int(f);
              asm volatile("cvt.rm.f32.s32 %0, %1;" : "=f"(f) : "r"((int)v7)); v7=__float_as_int(f); }
#elif OP == 3  // cvt.rni.s32.f32 (F2I round-nearest)
            { int ix; float f=__int_as_float(v0);
              asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(ix) : "f"(f)); v0=(unsigned)ix;
              asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v1))); v1=(unsigned)ix;
              asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v2))); v2=(unsigned)ix;
              asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v3))); v3=(unsigned)ix;
              asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v4))); v4=(unsigned)ix;
              asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v5))); v5=(unsigned)ix;
              asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v6))); v6=(unsigned)ix;
              asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v7))); v7=(unsigned)ix; }
#elif OP == 4  // cvt.rzi.s32.f32
            { int ix;
              asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v0))); v0=(unsigned)ix;
              asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v1))); v1=(unsigned)ix;
              asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v2))); v2=(unsigned)ix;
              asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v3))); v3=(unsigned)ix;
              asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v4))); v4=(unsigned)ix;
              asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v5))); v5=(unsigned)ix;
              asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v6))); v6=(unsigned)ix;
              asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(v7))); v7=(unsigned)ix; }
#elif OP == 5  // cvt.rn.sat.f32.f32 (FP32 saturate to [0,1])
            { float f;
              asm volatile("cvt.sat.f32.f32 %0, %1;" : "=f"(f) : "f"(__int_as_float(v0))); v0=__float_as_int(f);
              asm volatile("cvt.sat.f32.f32 %0, %1;" : "=f"(f) : "f"(__int_as_float(v1))); v1=__float_as_int(f);
              asm volatile("cvt.sat.f32.f32 %0, %1;" : "=f"(f) : "f"(__int_as_float(v2))); v2=__float_as_int(f);
              asm volatile("cvt.sat.f32.f32 %0, %1;" : "=f"(f) : "f"(__int_as_float(v3))); v3=__float_as_int(f);
              asm volatile("cvt.sat.f32.f32 %0, %1;" : "=f"(f) : "f"(__int_as_float(v4))); v4=__float_as_int(f);
              asm volatile("cvt.sat.f32.f32 %0, %1;" : "=f"(f) : "f"(__int_as_float(v5))); v5=__float_as_int(f);
              asm volatile("cvt.sat.f32.f32 %0, %1;" : "=f"(f) : "f"(__int_as_float(v6))); v6=__float_as_int(f);
              asm volatile("cvt.sat.f32.f32 %0, %1;" : "=f"(f) : "f"(__int_as_float(v7))); v7=__float_as_int(f); }
#elif OP == 6  // setp.eq.f16x2 (predicate from fp16 pair)
            {unsigned int p;
             asm volatile("{.reg .pred p0,p1; setp.eq.f16x2 p0|p1, %1, %2; selp.u32 %0, 1, 2, p0; }" : "=r"(p) : "r"(v0), "r"(v1));
             v0=p;
             asm volatile("{.reg .pred p0,p1; setp.eq.f16x2 p0|p1, %1, %2; selp.u32 %0, 1, 2, p0; }" : "=r"(p) : "r"(v2), "r"(v3));
             v2=p; }
#elif OP == 7  // set.eq.f16x2 (result = 0 or -1, not predicate)
            asm volatile("set.eq.f16x2.f16x2 %0, %1, %2;" : "=r"(x) : "r"(v0), "r"(v1)); v0=x;
            asm volatile("set.eq.f16x2.f16x2 %0, %1, %2;" : "=r"(x) : "r"(v2), "r"(v3)); v2=x;
            asm volatile("set.eq.f16x2.f16x2 %0, %1, %2;" : "=r"(x) : "r"(v4), "r"(v5)); v4=x;
            asm volatile("set.eq.f16x2.f16x2 %0, %1, %2;" : "=r"(x) : "r"(v6), "r"(v7)); v6=x;
#elif OP == 8  // cvt.rn.tf32.f32 (TF32 round)
            { unsigned int t;
              asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(t) : "f"(__int_as_float(v0))); v0=t;
              asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(t) : "f"(__int_as_float(v1))); v1=t;
              asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(t) : "f"(__int_as_float(v2))); v2=t;
              asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(t) : "f"(__int_as_float(v3))); v3=t;
              asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(t) : "f"(__int_as_float(v4))); v4=t;
              asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(t) : "f"(__int_as_float(v5))); v5=t;
              asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(t) : "f"(__int_as_float(v6))); v6=t;
              asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(t) : "f"(__int_as_float(v7))); v7=t; }
#elif OP == 9  // addc.u32 (add with carry)
            asm volatile("add.cc.u32 %0, %0, %1;" : "+r"(v0) : "r"(v1));
            asm volatile("addc.u32 %0, %0, %1;" : "+r"(v2) : "r"(v3));
            asm volatile("add.cc.u32 %0, %0, %1;" : "+r"(v4) : "r"(v5));
            asm volatile("addc.u32 %0, %0, %1;" : "+r"(v6) : "r"(v7));
#elif OP == 10 // LEA via mad.wide.u32 pattern (address compute)
            { unsigned long long w0, w1, w2, w3;
              asm volatile("mad.wide.u32 %0, %1, 4, %2;" : "=l"(w0) : "r"(v0), "l"((unsigned long long)A));
              asm volatile("mad.wide.u32 %0, %1, 4, %2;" : "=l"(w1) : "r"(v1), "l"((unsigned long long)A));
              asm volatile("mad.wide.u32 %0, %1, 4, %2;" : "=l"(w2) : "r"(v2), "l"((unsigned long long)A));
              asm volatile("mad.wide.u32 %0, %1, 4, %2;" : "=l"(w3) : "r"(v3), "l"((unsigned long long)A));
              v0 = (unsigned)(w0 ^ w1 ^ w2 ^ w3); }
#elif OP == 11 // selp.b32 data-dep
            { unsigned int p;
              asm volatile("{.reg .pred p; setp.lt.u32 p, %1, %2; selp.b32 %0, %1, %2, p;}" : "=r"(p) : "r"(v0), "r"(v1)); v0=p;
              asm volatile("{.reg .pred p; setp.lt.u32 p, %1, %2; selp.b32 %0, %1, %2, p;}" : "=r"(p) : "r"(v2), "r"(v3)); v2=p;
              asm volatile("{.reg .pred p; setp.lt.u32 p, %1, %2; selp.b32 %0, %1, %2, p;}" : "=r"(p) : "r"(v4), "r"(v5)); v4=p;
              asm volatile("{.reg .pred p; setp.lt.u32 p, %1, %2; selp.b32 %0, %1, %2, p;}" : "=r"(p) : "r"(v6), "r"(v7)); v6=p; }
#elif OP == 12 // bfi.b32
            asm volatile("bfi.b32 %0, %1, %2, 4, 8;" : "=r"(x) : "r"(v1), "r"(v0)); v0=x;
            asm volatile("bfi.b32 %0, %1, %2, 4, 8;" : "=r"(x) : "r"(v2), "r"(v3)); v2=x;
            asm volatile("bfi.b32 %0, %1, %2, 4, 8;" : "=r"(x) : "r"(v4), "r"(v5)); v4=x;
            asm volatile("bfi.b32 %0, %1, %2, 4, 8;" : "=r"(x) : "r"(v6), "r"(v7)); v6=x;
#elif OP == 13 // mul24.lo.u32 / .hi.u32 (24-bit mul)
            asm volatile("mul24.lo.u32 %0, %0, %1;" : "+r"(v0) : "r"(v1));
            asm volatile("mul24.lo.u32 %0, %0, %1;" : "+r"(v2) : "r"(v3));
            asm volatile("mul24.lo.u32 %0, %0, %1;" : "+r"(v4) : "r"(v5));
            asm volatile("mul24.lo.u32 %0, %0, %1;" : "+r"(v6) : "r"(v7));
#elif OP == 14 // mad24.lo.u32
            asm volatile("mad24.lo.u32 %0, %0, %1, %2;" : "+r"(v0) : "r"(v1), "r"(v2));
            asm volatile("mad24.lo.u32 %0, %0, %1, %2;" : "+r"(v3) : "r"(v4), "r"(v5));
            asm volatile("mad24.lo.u32 %0, %0, %1, %2;" : "+r"(v6) : "r"(v7), "r"(v0));
            asm volatile("mad24.lo.u32 %0, %0, %1, %2;" : "+r"(v1) : "r"(v2), "r"(v3));
#elif OP == 15 // red.shared.add.f32 (reduction, no return)
            { unsigned int base = (unsigned)__cvta_generic_to_shared(&smem[threadIdx.x]);
              asm volatile("red.shared.add.f32 [%0], %1;" :: "r"(base), "f"(__int_as_float(v0)));
              asm volatile("red.shared.add.f32 [%0], %1;" :: "r"(base), "f"(__int_as_float(v1)));
              asm volatile("red.shared.add.f32 [%0], %1;" :: "r"(base), "f"(__int_as_float(v2)));
              asm volatile("red.shared.add.f32 [%0], %1;" :: "r"(base), "f"(__int_as_float(v3)));
              asm volatile("red.shared.add.f32 [%0], %1;" :: "r"(base), "f"(__int_as_float(v4)));
              asm volatile("red.shared.add.f32 [%0], %1;" :: "r"(base), "f"(__int_as_float(v5)));
              asm volatile("red.shared.add.f32 [%0], %1;" :: "r"(base), "f"(__int_as_float(v6)));
              asm volatile("red.shared.add.f32 [%0], %1;" :: "r"(base), "f"(__int_as_float(v7))); }
#elif OP == 16 // shfl.sync.idx mask-all broadcast (already done, re-verify shape)
            asm volatile("shfl.sync.idx.b32 %0, %0, 0, 0x1F, -1;" : "+r"(v0));
            asm volatile("shfl.sync.idx.b32 %0, %0, 0, 0x1F, -1;" : "+r"(v1));
            asm volatile("shfl.sync.idx.b32 %0, %0, 0, 0x1F, -1;" : "+r"(v2));
            asm volatile("shfl.sync.idx.b32 %0, %0, 0, 0x1F, -1;" : "+r"(v3));
            asm volatile("shfl.sync.idx.b32 %0, %0, 0, 0x1F, -1;" : "+r"(v4));
            asm volatile("shfl.sync.idx.b32 %0, %0, 0, 0x1F, -1;" : "+r"(v5));
            asm volatile("shfl.sync.idx.b32 %0, %0, 0, 0x1F, -1;" : "+r"(v6));
            asm volatile("shfl.sync.idx.b32 %0, %0, 0, 0x1F, -1;" : "+r"(v7));
#elif OP == 17 // bar.cluster.arrive (CGA barrier — may require cluster)
            asm volatile("barrier.cluster.arrive.relaxed;");
            asm volatile("barrier.cluster.wait;");
#elif OP == 18 // mbarrier.arrive.shared (async barrier)
            { unsigned int base = (unsigned)__cvta_generic_to_shared(&smem[0]);
              unsigned long long tok;
              asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(tok) : "r"(base));
              v0 = (unsigned)tok; }
#endif
            v0 = x ^ v0;  // keep x live
        }
    }
    unsigned int acc = v0^v1^v2^v3^v4^v5^v6^v7;
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
