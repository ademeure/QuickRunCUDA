// §20 retest: pure FFMA2 baseline at varying ILP, plus FFMA2+ALU mixes.
//
// Catalog claim (B300_PIPE_CATALOG.md L7773+):
//   Pure FFMA2 ............... 5.57 cy/iter (baseline)
//   FFMA2 + 1 IADD ........... 6.76 cy/iter (+21%)
//   FFMA2 + 1 scalar FFMA .... 7.57 cy/iter (+36%)
//   FFMA2 + 2 FMIN ........... 9.45 cy/iter (+70%, "+35% per FMIN")
//
// User challenge: "5.57 cy/iter for pure FFMA2 is wrong; should be 4.0 (or 2.0
// at full ILP issue-rate-limited)".
//
// We measure cy/iter via clock64 inside a single-warp kernel and sweep N_CHAINS.
// A single inner iteration consists of:
//   for k in 0..N_CHAINS:   FFMA2 f[k] = f[k]*c1 + c0          (or other PATTERN)
//
// PATTERN selects the inner pattern (see #if blocks below).
// "cy/iter" = (total_clk - overhead) / (N_OUTER * N_INNER * N_CHAINS_USED)
// We report cy per ENTIRE inner-slot (= N_CHAINS * inst-per-chain) so it matches
// catalog cy/iter convention (cy per "row" of the table).

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef N_INNER
#define N_INNER 1024
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif
#ifndef PATTERN
#define PATTERN 0      // 0=pure FFMA2, 1=FFMA2+IADD, 2=FFMA2+scalarFFMA, 3=FFMA2+2FMIN
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int tid = threadIdx.x;

    // Per-chain FFMA2 accumulators (each holds one fp32x2 packed in u64).
    unsigned long long f[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        unsigned int ulo = __float_as_int(1.0001f + 0.0001f*(tid + k*23));
        unsigned int uhi = __float_as_int(1.0002f + 0.0001f*(tid + k*29));
        f[k] = ((unsigned long long)uhi << 32) | ulo;
    }
    unsigned int c1_u = __float_as_int(1.000001f);
    unsigned int c0_u = __float_as_int(0.9999f);
    unsigned long long c1 = ((unsigned long long)c1_u << 32) | c1_u;
    unsigned long long c0 = ((unsigned long long)c0_u << 32) | c0_u;

    // Per-chain side accumulators for the "+ALU" patterns.
    unsigned int u[N_CHAINS];
    float g[N_CHAINS];
    float scalar_b = 1.0000001f;
    float scalar_c = 0.0f;
    float fmin_const1 = 3.0f;
    float fmin_const2 = 4.0f;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        u[k] = (unsigned int)(tid * 7u + k * 13u + 0xab);
        g[k] = 1.0001f + 0.0001f*(tid + k*31);
    }

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                // FFMA2 always present.
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;"
                             : "+l"(f[k]) : "l"(c1), "l"(c0));
#if PATTERN == 1
                // + 1 IADD (chained on u[k])
                asm volatile("add.s32 %0, %0, %1;"
                             : "+r"(u[k]) : "r"(0x1234u));
#elif PATTERN == 2
                // + 1 scalar FFMA (chained on g[k])
                asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                             : "+f"(g[k]) : "f"(scalar_b), "f"(scalar_c));
#elif PATTERN == 3
                // + 2 FMIN (chained on g[k])
                asm volatile("min.f32 %0, %0, %1;"
                             : "+f"(g[k]) : "f"(fmin_const1));
                asm volatile("min.f32 %0, %0, %1;"
                             : "+f"(g[k]) : "f"(fmin_const2));
#endif
            }
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    // Anti-DCE: XOR everything, store under impossible predicate.
    unsigned long long acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        acc ^= f[k];
        acc ^= (unsigned long long)u[k];
        acc ^= (unsigned long long)__float_as_int(g[k]);
    }
    if (tid >= blockDim.x) {
        ((unsigned long long*)C)[blockIdx.x * blockDim.x + tid] = acc;
    }

    // Print cy/iter from one thread per block.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total_slots =
            (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
        unsigned long long total_clk = t1 - t0;
        // cy per "slot" (one slot = N_CHAINS chain-steps, where each chain-step
        // = 1 FFMA2 + the optional ALU op(s)). We report cy/slot/N_CHAINS so
        // the number is comparable to catalog "cy/iter" (per-chain per-row).
        // Catalog reports cy/iter = (cy per inner-loop iteration with their N_CHAINS).
        // We deliberately report TWO numbers: cy/inner-row (one chain-step) and cy/slot.
        double cy_per_chain_step = (double)total_clk / (double)(total_slots * N_CHAINS);
        double cy_per_slot = (double)total_clk / (double)total_slots;
        printf("PATTERN=%d N_CHAINS=%d N_INNER=%d N_OUTER=%d total_clk=%llu cy/chain_step=%.4f cy/slot=%.4f\n",
               PATTERN, N_CHAINS, N_INNER, N_OUTER, total_clk,
               cy_per_chain_step, cy_per_slot);
    }
}
