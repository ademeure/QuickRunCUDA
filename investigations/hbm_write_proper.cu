// hbm_write_proper.cu
// TRUE HBM write bandwidth characterization for B300 SXM6
//
// Strategy:
//   - Working set >= 4 GB (well past L2 capacity of ~126 MB) to guarantee DRAM-bound
//   - Grid-stride loop across the working set
//   - Anti-DCE: write values derived from thread ID + loop iteration (not constants)
//   - Test scalar (b32), v4 (b32x4=16B), v8 (b32x8=32B), plus cache hints
//   - Separate OP codes selectable via -H "#define OP N"
//
// OP codes:
//   0 = st.global.b32        (scalar, 4 B/thread/iter)
//   1 = st.global.v4.b32     (v4, 16 B/thread/iter)
//   2 = st.global.v8.b32     (v8, 32 B/thread/iter)
//   3 = st.global.cs.b32     (scalar + streaming hint)
//   4 = st.global.cs.v4.b32  (v4 + streaming hint)
//   5 = st.global.wb.b32     (scalar + write-back hint)
//   6 = st.global.wb.v4.b32  (v4 + write-back hint)
//   7 = st.volatile.global.b32 (volatile scalar)
//   8 = st.global.cg.v4.b32  (v4 + cache-global hint, L1 bypass)

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
