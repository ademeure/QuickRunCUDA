// hbm_write_proper.cu
// TRUE HBM write bandwidth characterization for B300 SXM6
// See: investigations/08_hbm_write.md for full analysis
//
// RESULT SUMMARY (ncu-verified, 2026-04-17):
//   True HBM write peak: 6.1-6.3 TB/s (all variants)
//   True HBM read  peak: 6.85-6.88 TB/s (for comparison)
//   Write is ~9% slower than read (HBM3E asymmetry)
//
// WARNING: Using QuickRunCUDA's warmup loop gives ~12 TB/s "fire-and-forget" artifact.
//   Use standalone binary + ncu dram__bytes_write.sum.per_second for ground truth.
//
// Strategy:
//   - Working set >= 4 GB (well past L2 capacity of ~126 MB) to guarantee DRAM-bound
//   - Grid-stride loop across the working set
//   - Anti-DCE: write values derived from thread ID + loop iteration (not constants)
//   - Test scalar (b32), v4 (b32x4=16B), v8 (b32x8=32B), plus cache hints
//   - Separate OP codes selectable via -H "#define OP N"
//
// OP codes:
//   0 = st.global.b32        (scalar, 4 B/thread/iter)   -> STG.E
//   1 = st.global.v4.b32     (v4, 16 B/thread/iter)      -> STG.E.128
//   2 = st.global.v8.b32     (v8, 32 B/thread/iter)      -> STG.E.ENL2.256
//   3 = st.global.cs.b32     (scalar + streaming hint)   -> STG.E.EF
//   4 = st.global.cs.v4.b32  (v4 + streaming hint)       -> STG.E.EF.128
//   5 = st.global.wb.b32     (scalar + write-back hint)  -> STG.E.STRONG.SM
//   6 = st.global.wb.v4.b32  (v4 + write-back hint)      -> STG.E.STRONG.SM.128
//   7 = st.volatile.global.b32 (volatile scalar)         -> STG.E (volatile)
//   8 = st.global.cg.v4.b32  (v4 + cache-global hint)   -> STG.E.128 (L1 bypass)
//
// Run with (recommended: ncu for ground truth, not QuickRunCUDA timing):
//   # 4 GB buffer, 2 CTAs/SM x 256 threads = 75776 threads
//   # OP 0,3,5,7 (4B): ITERS=14168; OP 1,4,6,8 (16B): ITERS=3536; OP 2 (32B): ITERS=1768
//   ./QuickRunCUDA investigations/hbm_write_proper.cu \
//     -H "#define OP 4" -p -t 256 -b 2 -T 7 -B 1073741824 -0 3536

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef UNROLL
#define UNROLL 8
#endif

#ifndef OP
#define OP 1
#endif

// Bytes per thread per store instruction (BPT)
// OP 0,3,5,7 -> 4 B; OP 1,4,6,8 -> 16 B; OP 2 -> 32 B
#if OP == 2
  #define BPT 32
#elif OP == 0
  #define BPT 4
#elif OP == 3
  #define BPT 4
#elif OP == 5
  #define BPT 4
#elif OP == 7
  #define BPT 4
#else
  // OP 1,4,6,8
  #define BPT 16
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C,
            int ITERS, int arg1, int arg2)
{
    // B is the write target.
    // Grid-stride pattern: each thread writes to address base + (tid + lap*nthreads)*BPT
    unsigned long long n_threads = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long tid       = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Anti-DCE checksum written to C at end
    unsigned ck = (unsigned)tid;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned long long byte_off =
                (tid + (unsigned long long)(i + j) * n_threads) * (unsigned long long)BPT;

            unsigned long long addr = (unsigned long long)B + byte_off;
            unsigned v = (unsigned)(tid ^ (unsigned long long)(i + j));

#if OP == 0
            asm volatile("st.global.b32 [%0], %1;"
                :: "l"(addr), "r"(v) : "memory");
            ck ^= v;
#elif OP == 1
            asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};"
                :: "l"(addr), "r"(v), "r"(v+1u), "r"(v+2u), "r"(v+3u) : "memory");
            ck ^= v;
#elif OP == 2
            asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                :: "l"(addr),
                   "r"(v), "r"(v+1u), "r"(v+2u), "r"(v+3u),
                   "r"(v+4u), "r"(v+5u), "r"(v+6u), "r"(v+7u)
                : "memory");
            ck ^= v;
#elif OP == 3
            asm volatile("st.global.cs.b32 [%0], %1;"
                :: "l"(addr), "r"(v) : "memory");
            ck ^= v;
#elif OP == 4
            asm volatile("st.global.cs.v4.b32 [%0], {%1,%2,%3,%4};"
                :: "l"(addr), "r"(v), "r"(v+1u), "r"(v+2u), "r"(v+3u) : "memory");
            ck ^= v;
#elif OP == 5
            asm volatile("st.global.wb.b32 [%0], %1;"
                :: "l"(addr), "r"(v) : "memory");
            ck ^= v;
#elif OP == 6
            asm volatile("st.global.wb.v4.b32 [%0], {%1,%2,%3,%4};"
                :: "l"(addr), "r"(v), "r"(v+1u), "r"(v+2u), "r"(v+3u) : "memory");
            ck ^= v;
#elif OP == 7
            asm volatile("st.volatile.global.b32 [%0], %1;"
                :: "l"(addr), "r"(v) : "memory");
            ck ^= v;
#elif OP == 8
            asm volatile("st.global.cg.v4.b32 [%0], {%1,%2,%3,%4};"
                :: "l"(addr), "r"(v), "r"(v+1u), "r"(v+2u), "r"(v+3u) : "memory");
            ck ^= v;
#endif
        }
    }

    // Write checksum to prevent DCE of entire loop
    if (ck == 0xDEADBEEF) C[tid % 1024u] = ck;
}
