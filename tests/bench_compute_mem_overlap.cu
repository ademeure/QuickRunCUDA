// §22h compute-memory overlap microbench
// Single-thread (tid=0, blk=0) to get clean clock64 timing.
// Inner loop: 1 cold-DRAM load (.cg, L1-bypass) + N_FFMA independent FFMA chains.
// Vary N_FFMA to find when FFMAs stop being hidden by memory latency.
//
// Args:
//   arg0 = outer_iters (timed iteration count of the inner loop)
//   arg1 = stride_in_dwords (used to defeat hardware prefetcher; 0 -> default)
//   arg2 = anti-DCE seed (impossible-match value, but stored anyway)
//
// Output written to A[0]: cycle_count (long long).

#ifndef N_FFMA
#define N_FFMA 0
#endif

#ifndef OUTER_ITERS_DEFAULT
#define OUTER_ITERS_DEFAULT 4096
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C,
                                  int outer_iters, int arg1, int arg2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (outer_iters <= 0) outer_iters = OUTER_ITERS_DEFAULT;

    int *p = (int*)A;
    // Working set: A is allocated 64M dwords = 256 MiB by default. We index
    // with a pseudo-random walk that touches the entire working set, so each
    // load is expected to miss L1, L2, and L1.5 (texture) — true cold DRAM.
    //
    // Walk: v <- v * 1664525 + 1013904223 (Numerical Recipes LCG).
    // Mask to (64M - 1) dwords = 0x03FFFFFF -> spans all 256 MiB.
    unsigned int v = (unsigned int)(threadIdx.x + 12345);

    // FFMA chains: each in its own register, all using shared coeffs.
    // float chain[N_FFMA] would spill; declare individuals.
    float coef_m = 1.0000001f;
    float coef_a = 0.9999999f;
#if N_FFMA >= 1
    float r0  = (float)threadIdx.x + 0.0f;
#endif
#if N_FFMA >= 2
    float r1  = (float)threadIdx.x + 1.0f;
#endif
#if N_FFMA >= 4
    float r2  = (float)threadIdx.x + 2.0f;
    float r3  = (float)threadIdx.x + 3.0f;
#endif
#if N_FFMA >= 8
    float r4  = (float)threadIdx.x + 4.0f;
    float r5  = (float)threadIdx.x + 5.0f;
    float r6  = (float)threadIdx.x + 6.0f;
    float r7  = (float)threadIdx.x + 7.0f;
#endif
#if N_FFMA >= 16
    float r8  = (float)threadIdx.x + 8.0f;
    float r9  = (float)threadIdx.x + 9.0f;
    float r10 = (float)threadIdx.x + 10.0f;
    float r11 = (float)threadIdx.x + 11.0f;
    float r12 = (float)threadIdx.x + 12.0f;
    float r13 = (float)threadIdx.x + 13.0f;
    float r14 = (float)threadIdx.x + 14.0f;
    float r15 = (float)threadIdx.x + 15.0f;
#endif
#if N_FFMA >= 32
    float r16 = 16.f, r17 = 17.f, r18 = 18.f, r19 = 19.f;
    float r20 = 20.f, r21 = 21.f, r22 = 22.f, r23 = 23.f;
    float r24 = 24.f, r25 = 25.f, r26 = 26.f, r27 = 27.f;
    float r28 = 28.f, r29 = 29.f, r30 = 30.f, r31 = 31.f;
#endif
#if N_FFMA >= 48
    float r32 = 32.f, r33 = 33.f, r34 = 34.f, r35 = 35.f;
    float r36 = 36.f, r37 = 37.f, r38 = 38.f, r39 = 39.f;
    float r40 = 40.f, r41 = 41.f, r42 = 42.f, r43 = 43.f;
    float r44 = 44.f, r45 = 45.f, r46 = 46.f, r47 = 47.f;
#endif
#if N_FFMA >= 64
    float r48 = 48.f, r49 = 49.f, r50 = 50.f, r51 = 51.f;
    float r52 = 52.f, r53 = 53.f, r54 = 54.f, r55 = 55.f;
    float r56 = 56.f, r57 = 57.f, r58 = 58.f, r59 = 59.f;
    float r60 = 60.f, r61 = 61.f, r62 = 62.f, r63 = 63.f;
#endif
#if N_FFMA >= 96
    float r64 = 64.f, r65 = 65.f, r66 = 66.f, r67 = 67.f;
    float r68 = 68.f, r69 = 69.f, r70 = 70.f, r71 = 71.f;
    float r72 = 72.f, r73 = 73.f, r74 = 74.f, r75 = 75.f;
    float r76 = 76.f, r77 = 77.f, r78 = 78.f, r79 = 79.f;
    float r80 = 80.f, r81 = 81.f, r82 = 82.f, r83 = 83.f;
    float r84 = 84.f, r85 = 85.f, r86 = 86.f, r87 = 87.f;
    float r88 = 88.f, r89 = 89.f, r90 = 90.f, r91 = 91.f;
    float r92 = 92.f, r93 = 93.f, r94 = 94.f, r95 = 95.f;
#endif
#if N_FFMA >= 128
    float r96  = 96.f, r97  = 97.f, r98  = 98.f, r99  = 99.f;
    float r100 = 100.f, r101 = 101.f, r102 = 102.f, r103 = 103.f;
    float r104 = 104.f, r105 = 105.f, r106 = 106.f, r107 = 107.f;
    float r108 = 108.f, r109 = 109.f, r110 = 110.f, r111 = 111.f;
    float r112 = 112.f, r113 = 113.f, r114 = 114.f, r115 = 115.f;
    float r116 = 116.f, r117 = 117.f, r118 = 118.f, r119 = 119.f;
    float r120 = 120.f, r121 = 121.f, r122 = 122.f, r123 = 123.f;
    float r124 = 124.f, r125 = 125.f, r126 = 126.f, r127 = 127.f;
#endif

    long long t0 = clock64();

    #pragma unroll 1
    for (int it = 0; it < outer_iters; it++) {
        // pseudo-random walk: defeats hw prefetcher, spans 256 MiB
        v = v * 1664525u + 1013904223u;
        unsigned int idx = v & 0x03FFFFFFu;  // 64M dwords = 256 MiB
        int loaded;
        // .cg: cache global (skips L1, hits L2 + DRAM). On Hopper/Blackwell,
        // .cg is the canonical "cold-DRAM probe" hint.
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(loaded) : "l"(p + idx));
        v ^= (unsigned int)loaded;

        // Independent FFMA chains: each uses its own register so they're
        // truly parallel (4-way ILP per SMSP, no chain dep).
#if N_FFMA >= 1
        r0 = r0 * coef_m + coef_a;
#endif
#if N_FFMA >= 2
        r1 = r1 * coef_m + coef_a;
#endif
#if N_FFMA >= 4
        r2 = r2 * coef_m + coef_a;
        r3 = r3 * coef_m + coef_a;
#endif
#if N_FFMA >= 8
        r4 = r4 * coef_m + coef_a;
        r5 = r5 * coef_m + coef_a;
        r6 = r6 * coef_m + coef_a;
        r7 = r7 * coef_m + coef_a;
#endif
#if N_FFMA >= 16
        r8  = r8  * coef_m + coef_a;
        r9  = r9  * coef_m + coef_a;
        r10 = r10 * coef_m + coef_a;
        r11 = r11 * coef_m + coef_a;
        r12 = r12 * coef_m + coef_a;
        r13 = r13 * coef_m + coef_a;
        r14 = r14 * coef_m + coef_a;
        r15 = r15 * coef_m + coef_a;
#endif
#if N_FFMA >= 32
        r16 = r16 * coef_m + coef_a; r17 = r17 * coef_m + coef_a;
        r18 = r18 * coef_m + coef_a; r19 = r19 * coef_m + coef_a;
        r20 = r20 * coef_m + coef_a; r21 = r21 * coef_m + coef_a;
        r22 = r22 * coef_m + coef_a; r23 = r23 * coef_m + coef_a;
        r24 = r24 * coef_m + coef_a; r25 = r25 * coef_m + coef_a;
        r26 = r26 * coef_m + coef_a; r27 = r27 * coef_m + coef_a;
        r28 = r28 * coef_m + coef_a; r29 = r29 * coef_m + coef_a;
        r30 = r30 * coef_m + coef_a; r31 = r31 * coef_m + coef_a;
#endif
#if N_FFMA >= 48
        r32 = r32 * coef_m + coef_a; r33 = r33 * coef_m + coef_a;
        r34 = r34 * coef_m + coef_a; r35 = r35 * coef_m + coef_a;
        r36 = r36 * coef_m + coef_a; r37 = r37 * coef_m + coef_a;
        r38 = r38 * coef_m + coef_a; r39 = r39 * coef_m + coef_a;
        r40 = r40 * coef_m + coef_a; r41 = r41 * coef_m + coef_a;
        r42 = r42 * coef_m + coef_a; r43 = r43 * coef_m + coef_a;
        r44 = r44 * coef_m + coef_a; r45 = r45 * coef_m + coef_a;
        r46 = r46 * coef_m + coef_a; r47 = r47 * coef_m + coef_a;
#endif
#if N_FFMA >= 64
        r48 = r48 * coef_m + coef_a; r49 = r49 * coef_m + coef_a;
        r50 = r50 * coef_m + coef_a; r51 = r51 * coef_m + coef_a;
        r52 = r52 * coef_m + coef_a; r53 = r53 * coef_m + coef_a;
        r54 = r54 * coef_m + coef_a; r55 = r55 * coef_m + coef_a;
        r56 = r56 * coef_m + coef_a; r57 = r57 * coef_m + coef_a;
        r58 = r58 * coef_m + coef_a; r59 = r59 * coef_m + coef_a;
        r60 = r60 * coef_m + coef_a; r61 = r61 * coef_m + coef_a;
        r62 = r62 * coef_m + coef_a; r63 = r63 * coef_m + coef_a;
#endif
#if N_FFMA >= 96
        r64 = r64 * coef_m + coef_a; r65 = r65 * coef_m + coef_a;
        r66 = r66 * coef_m + coef_a; r67 = r67 * coef_m + coef_a;
        r68 = r68 * coef_m + coef_a; r69 = r69 * coef_m + coef_a;
        r70 = r70 * coef_m + coef_a; r71 = r71 * coef_m + coef_a;
        r72 = r72 * coef_m + coef_a; r73 = r73 * coef_m + coef_a;
        r74 = r74 * coef_m + coef_a; r75 = r75 * coef_m + coef_a;
        r76 = r76 * coef_m + coef_a; r77 = r77 * coef_m + coef_a;
        r78 = r78 * coef_m + coef_a; r79 = r79 * coef_m + coef_a;
        r80 = r80 * coef_m + coef_a; r81 = r81 * coef_m + coef_a;
        r82 = r82 * coef_m + coef_a; r83 = r83 * coef_m + coef_a;
        r84 = r84 * coef_m + coef_a; r85 = r85 * coef_m + coef_a;
        r86 = r86 * coef_m + coef_a; r87 = r87 * coef_m + coef_a;
        r88 = r88 * coef_m + coef_a; r89 = r89 * coef_m + coef_a;
        r90 = r90 * coef_m + coef_a; r91 = r91 * coef_m + coef_a;
        r92 = r92 * coef_m + coef_a; r93 = r93 * coef_m + coef_a;
        r94 = r94 * coef_m + coef_a; r95 = r95 * coef_m + coef_a;
#endif
#if N_FFMA >= 128
        r96  = r96  * coef_m + coef_a; r97  = r97  * coef_m + coef_a;
        r98  = r98  * coef_m + coef_a; r99  = r99  * coef_m + coef_a;
        r100 = r100 * coef_m + coef_a; r101 = r101 * coef_m + coef_a;
        r102 = r102 * coef_m + coef_a; r103 = r103 * coef_m + coef_a;
        r104 = r104 * coef_m + coef_a; r105 = r105 * coef_m + coef_a;
        r106 = r106 * coef_m + coef_a; r107 = r107 * coef_m + coef_a;
        r108 = r108 * coef_m + coef_a; r109 = r109 * coef_m + coef_a;
        r110 = r110 * coef_m + coef_a; r111 = r111 * coef_m + coef_a;
        r112 = r112 * coef_m + coef_a; r113 = r113 * coef_m + coef_a;
        r114 = r114 * coef_m + coef_a; r115 = r115 * coef_m + coef_a;
        r116 = r116 * coef_m + coef_a; r117 = r117 * coef_m + coef_a;
        r118 = r118 * coef_m + coef_a; r119 = r119 * coef_m + coef_a;
        r120 = r120 * coef_m + coef_a; r121 = r121 * coef_m + coef_a;
        r122 = r122 * coef_m + coef_a; r123 = r123 * coef_m + coef_a;
        r124 = r124 * coef_m + coef_a; r125 = r125 * coef_m + coef_a;
        r126 = r126 * coef_m + coef_a; r127 = r127 * coef_m + coef_a;
#endif
    }

    long long t1 = clock64();

    // Anti-DCE: store to C if impossible-true (compiler can't prove false).
    if (arg2 == 999999) {
        ((int*)C)[0] = (int)v;
        float sum = 0.f;
#if N_FFMA >= 1
        sum += r0;
#endif
#if N_FFMA >= 2
        sum += r1;
#endif
#if N_FFMA >= 4
        sum += r2 + r3;
#endif
#if N_FFMA >= 8
        sum += r4 + r5 + r6 + r7;
#endif
#if N_FFMA >= 16
        sum += r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
#endif
#if N_FFMA >= 32
        sum += r16+r17+r18+r19+r20+r21+r22+r23+r24+r25+r26+r27+r28+r29+r30+r31;
#endif
#if N_FFMA >= 48
        sum += r32+r33+r34+r35+r36+r37+r38+r39+r40+r41+r42+r43+r44+r45+r46+r47;
#endif
#if N_FFMA >= 64
        sum += r48+r49+r50+r51+r52+r53+r54+r55+r56+r57+r58+r59+r60+r61+r62+r63;
#endif
#if N_FFMA >= 96
        sum += r64+r65+r66+r67+r68+r69+r70+r71+r72+r73+r74+r75+r76+r77+r78+r79
             + r80+r81+r82+r83+r84+r85+r86+r87+r88+r89+r90+r91+r92+r93+r94+r95;
#endif
#if N_FFMA >= 128
        sum += r96+r97+r98+r99+r100+r101+r102+r103+r104+r105+r106+r107+r108+r109+r110+r111
             + r112+r113+r114+r115+r116+r117+r118+r119+r120+r121+r122+r123+r124+r125+r126+r127;
#endif
        ((float*)C)[1] = sum;
    }

    long long delta = t1 - t0;
    *((long long*)A) = delta;
    // Also report
    printf("N_FFMA=%d outer_iters=%d cycles=%lld cy/iter=%.3f\n",
           N_FFMA, outer_iters, delta, (double)delta / (double)outer_iters);
}
