// Port-hypothesis discrimination tests.
//
// VARIANT:
//   10 = pack narrow (UNPACK_B_MERGE_C) with *independent* dest regs (no feedback)
//        — if still 32/clk, MERGE_C's dest-read is NOT the throttle (rules out pure hypothesis 1-MERGEC)
//
//   11 = PACK_AB f32,f32->f16x2 with *independent* dest regs (no feedback)
//        — no MERGE_C at all; verifies 2-source PACK still 32/clk independent of feedback
//
//   12 = UNPACK_B (no merge) with feedback → destination SAME as source (force "RMW-like")
//        — tests if it's dest-read or simply port count. Should still be 64/clk.
//
//   13 = Two parallel pack-narrow chains, writing to TWO different destination regs per
//        cycle (writes to R_dst_a, R_dst_b). Probes whether 2 packs could dual-issue
//        if their writebacks were guaranteed non-conflicting. If 32/clk persists → rules
//        out writeback-bus conflict (hypothesis 2).
//
//   14 = PACK_AB f32 pair -> tf32 (PACK_B) with 2 different f32 inputs manually
//        combined (tests if true 1R pipelines dual-issue regardless of writeback).
//        Should be 64/clk.
//
//   15 = Mix: one pack (3R) + one unpack (1R) per "cycle" — should saturate at 64 total
//        (1 slot each). Baseline confirmation.
//
//   16 = Pack-narrow with src and dst on TWO DIFFERENT physical registers forced via
//        inline asm (no possibility of reg reallocation by ptxas).

#ifndef VARIANT
#define VARIANT 10
#endif
#ifndef N_CHAINS
#define N_CHAINS 16
#endif
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {

    // All variants use N_CHAINS independent inputs
    unsigned int src[N_CHAINS];
    unsigned int dst[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        src[k] = 0x3C003C01u ^ (threadIdx.x + k);
        dst[k] = 0xDEAD0000u ^ (threadIdx.x + k);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if VARIANT == 10
                // PACK narrow, but force SASS to use Rc = real register (not RZ) by
                // merging the narrow result into the upper half of a 32-bit word that
                // holds loop-dependent state (dst[k]). PTX '.b32 pack' semantics will
                // cause ptxas to emit F2FP where Rc is actually read.
                //   dst[k] = (dst[k] & 0xFFFF0000) | narrow(src[k])
                unsigned short h;
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                             : "=h"(h) : "r"(src[k]));
                // Merge narrow result into high half of dst (this should become a MERGE
                // target if ptxas is clever, or an IADD/LOP3; either way, src mutates)
                dst[k] = (dst[k] & 0xFFFF0000u) | (unsigned int)h;
                src[k] = src[k] ^ dst[k];  // feedback so can't hoist

#elif VARIANT == 11
                // PACK_AB (no MERGE_C). Two independent f32 sources, write to full 32-bit
                // destination that is different from both sources.
                float f_lo = __int_as_float(src[k]);
                float f_hi = __int_as_float(dst[k]);
                unsigned int out;
                asm volatile("cvt.rn.f16x2.f32 %0, %2, %1;"
                             : "=r"(out) : "f"(f_lo), "f"(f_hi));
                dst[k] ^= out;

#elif VARIANT == 12
                // UNPACK with tight feedback RMW, keep output going back to source register.
                // Tests: if "dest-read" were the issue, forcing feedback would slow us down.
                unsigned short hin = (unsigned short)src[k];
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                             : "=r"(src[k]) : "h"(hin));

#elif VARIANT == 13
                // Two parallel pack-narrow chains writing to separate destinations.
                // Even though each chain's MERGE_C is independent, both need regfile
                // read ports. If writeback-bus were the bottleneck (hypothesis 2),
                // these pairs should dual-issue.
                unsigned short h1, h2;
                // both packs write distinct 16-bit destinations — no write conflict
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                             : "=h"(h1) : "r"(src[k]));
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                             : "=h"(h2) : "r"(dst[k]));
                src[k] = (unsigned int)h1;
                dst[k] = (unsigned int)h2;

#elif VARIANT == 14
                // Scalar tf32 PACK_B baseline (should be 64/clk).
                float f = __int_as_float(src[k]);
                unsigned int out;
                asm volatile("cvt.rn.satfinite.tf32.f32 %0, %1;"
                             : "=r"(out) : "f"(f));
                src[k] = out;

#elif VARIANT == 15
                // Round-trip: one pack + one unpack. Baseline to confirm 2-slot model.
                unsigned short hpack;
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                             : "=h"(hpack) : "r"(src[k]));
                unsigned int unpacked;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                             : "=r"(unpacked) : "h"(hpack));
                src[k] = unpacked;

#elif VARIANT == 16
                // PACK_AB with feedback so ptxas can't hoist.
                float f_lo = __int_as_float(src[k]);
                float f_hi = __int_as_float(dst[k]);
                unsigned int out;
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;"
                             : "=r"(out) : "f"(f_lo), "f"(f_hi));
                src[k] = out;
                dst[k] = dst[k] ^ out;

#elif VARIANT == 17
                // Four independent pack-narrow chains per inner iteration. Maximum
                // register pressure, maximum ILP. If a magical configuration lets
                // two packs dual-issue, 4-wide would surface it.
                unsigned short h1, h2, h3, h4;
                unsigned int s1 = src[k];
                unsigned int s2 = dst[k];
                unsigned int s3 = src[(k+1) % N_CHAINS];
                unsigned int s4 = dst[(k+1) % N_CHAINS];
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(h1) : "r"(s1));
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(h2) : "r"(s2));
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(h3) : "r"(s3));
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(h4) : "r"(s4));
                src[k] = ((unsigned int)h1) | (((unsigned int)h2) << 16);
                dst[k] = ((unsigned int)h3) | (((unsigned int)h4) << 16);

#elif VARIANT == 18
                // Pack using the MOV-merge pattern explicitly: produce narrow,
                // merge into dst[k] via LOP3 — confirms 2R pack issue rate, and
                // the overhead doesn't hide a "real" dual-issue pack.
                unsigned short h;
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                             : "=h"(h) : "r"(src[k]));
                dst[k] ^= (unsigned int)h;
                src[k] = src[k] ^ dst[k];

#elif VARIANT == 19
                // Control: pure UNPACK_B 4-wide (4 independent unpacks per k).
                // Should approach 64/SM/clk since 4× ILP is more than enough.
                unsigned short h1 = (unsigned short)src[k];
                unsigned short h2 = (unsigned short)(src[k] >> 16);
                unsigned short h3 = (unsigned short)dst[k];
                unsigned short h4 = (unsigned short)(dst[k] >> 16);
                unsigned int u1, u2, u3, u4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(u1) : "h"(h1));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(u2) : "h"(h2));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(u3) : "h"(h3));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(u4) : "h"(h4));
                src[k] = u1 ^ u2;
                dst[k] = u3 ^ u4;
#endif
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= src[k] ^ dst[k];

    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
