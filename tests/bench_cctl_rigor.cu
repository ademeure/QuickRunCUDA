// Rigorous CCTL.IVALL cost vs L1-resident lines.
// FIXES the prior less-rigorous test:
//   - Set L1 carveout to MAX-L1 (via cudaFuncSetAttribute, but here via PTX setmaxnreg or
//     via host-side; QuickRunCUDA doesn't expose carveout, so we use cudaDevAttrMaxSmemPerBlock
//     option to coerce a small smem budget = max L1. WORKAROUND: declare a small smem array
//     so the dynamic smem budget is small.
//   - Use nanosleep AFTER the fill, BEFORE the timed CCTL, to ensure all loads are
//     FULLY drained and the CCTL cost is purely invalidation, not drain wait.
//   - Probe multiple L1 fill amounts.
//
// MODE selects test variant:
//   0 = empty baseline (clock64 noise floor)
//   1 = nanosleep only (verify nanosleep cost itself)
//   2 = lone CCTL (after nanosleep)
//   3 = fill K KB cached loads → nanosleep → 1 CCTL
//   4 = MODE 3 then a second CCTL (L1 should be empty for second)

#ifndef MODE
#define MODE 3
#endif
#ifndef N_FILL_KB
#define N_FILL_KB 16
#endif
#ifndef SLEEP_NS
#define SLEEP_NS 10000
#endif
#ifndef N_OUTER
#define N_OUTER 30
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
#ifndef WARP_TEST
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
#else
    if (blockIdx.x != 0) return;  // full warp from block 0
#endif

    int* workspace = (int*)A;
    const int WS_INTS = N_FILL_KB * 256;  // KB → ints (1 KB = 256 ints)
    int v = (int)threadIdx.x + 1;

    unsigned long long t0 = 0, t1 = 0;
    long long total_dt = 0;

    asm volatile("fence.acquire.gpu;" ::: "memory");
    // Warm: nanosleep so we start each timed iteration in a clean state
    asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));

    for (int it = 0; it < N_OUTER; it++) {
#if MODE == 0
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);

#elif MODE == 1
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);

#elif MODE == 2
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));  // ensure idle
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");  // CCTL.IVALL on truly-empty L1
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);

#elif MODE == 3
        // Fill L1 with cached loads
        int fill_v = v;
        for (int j = 0; j < WS_INTS; j += 32) {
            int loaded;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(loaded) : "l"(workspace + j));
            fill_v ^= loaded;
        }
        v ^= fill_v;
        // Drain wait: nanosleep so all loads are fully complete in L1
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        // Timed CCTL
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);

#elif MODE == 10  // MEMBAR.ALL.GPU on truly-idle pipeline (fence.release.gpu)
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 11  // MEMBAR.ALL.SYS (fence.release.sys) on idle pipeline
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.sys;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 12  // fence.acq_rel.gpu = MEMBAR + CCTL on idle
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 13  // MEMBAR.ALL.GPU after writes (with drain) — see if writes drain through MEMBAR is faster than naked
        for (int j = 0; j < WS_INTS; j += 32) {
            int rval = v + j;
            asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + j), "r"(rval) : "memory");
        }
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 30  // ONE load just before CCTL (no drain) — measures CCTL = drain hypothesis
        // Issue one load with chain dependency to force completion before t0
        int loaded;
        // Use chain-dep address so the load can't be reordered earlier
        unsigned int addr_off = ((unsigned)v ^ (unsigned)u1) & 0x3FFFFu;  // small WS = L1/L2-hit
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(loaded) : "l"(workspace + addr_off));
        // CRUCIAL: t0 is read AFTER load issue but the load itself may still be in flight
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        // Use the loaded value to force load completion BEFORE CCTL
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded;  // anti-DCE
        total_dt += (long long)(t1 - t0);
#elif MODE == 31  // ONE load LARGE WS (DRAM-bound) just before CCTL
        int loaded;
        // Use BIG WS so load is cold-DRAM
        unsigned int addr_off = ((unsigned)v * 0x9E3779B1u) & 0x0FFFFFFFu;  // 1 GB span
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(loaded) : "l"(workspace + addr_off));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded;
        total_dt += (long long)(t1 - t0);
#elif MODE == 32  // ONE load L1-resident (warm small WS) before CCTL
        int loaded;
        // Pre-warm: load same addr first
        unsigned int addr_off = 0;
        for (int p = 0; p < 8; p++) {
            int prewarm;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(prewarm) : "l"(workspace + addr_off));
            v ^= prewarm;
        }
        // Now timed: a single L1-hit load, then CCTL
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(loaded) : "l"(workspace + addr_off));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded;
        total_dt += (long long)(t1 - t0);

#elif MODE == 40  // 8 cold-DRAM loads + CCTL (amplify drain wait)
        // Stride pattern guaranteed to miss L1/L2 (1 GB span, 4 KB stride)
        int sum_d = 0;
        unsigned int base_d = ((unsigned)v * 0x9E3779B1u) & 0x0FFFFFFFu;
        int ld40_0, ld40_1, ld40_2, ld40_3, ld40_4, ld40_5, ld40_6, ld40_7;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(ld40_0) : "l"(workspace + ((base_d + 0*1024) & 0x0FFFFFFFu)));
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(ld40_1) : "l"(workspace + ((base_d + 1*1024) & 0x0FFFFFFFu)));
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(ld40_2) : "l"(workspace + ((base_d + 2*1024) & 0x0FFFFFFFu)));
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(ld40_3) : "l"(workspace + ((base_d + 3*1024) & 0x0FFFFFFFu)));
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(ld40_4) : "l"(workspace + ((base_d + 4*1024) & 0x0FFFFFFFu)));
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(ld40_5) : "l"(workspace + ((base_d + 5*1024) & 0x0FFFFFFFu)));
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(ld40_6) : "l"(workspace + ((base_d + 6*1024) & 0x0FFFFFFFu)));
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(ld40_7) : "l"(workspace + ((base_d + 7*1024) & 0x0FFFFFFFu)));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        sum_d = ld40_0 ^ ld40_1 ^ ld40_2 ^ ld40_3 ^ ld40_4 ^ ld40_5 ^ ld40_6 ^ ld40_7;
        v ^= sum_d;
        total_dt += (long long)(t1 - t0);

#elif MODE == 41  // 8 L1-resident loads + CCTL (small WS, pre-warmed)
        // Pre-warm 8 distinct lines (8*128 = 1 KB)
        int prew_0=0, prew_1=0, prew_2=0, prew_3=0, prew_4=0, prew_5=0, prew_6=0, prew_7=0;
        for (int p = 0; p < 4; p++) {
            int x;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(workspace + 0));   prew_0 ^= x;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(workspace + 32));  prew_1 ^= x;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(workspace + 64));  prew_2 ^= x;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(workspace + 96));  prew_3 ^= x;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(workspace + 128)); prew_4 ^= x;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(workspace + 160)); prew_5 ^= x;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(workspace + 192)); prew_6 ^= x;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(workspace + 224)); prew_7 ^= x;
        }
        v ^= prew_0 ^ prew_1 ^ prew_2 ^ prew_3 ^ prew_4 ^ prew_5 ^ prew_6 ^ prew_7;
        // Timed: 8 L1-hit loads, then CCTL
        int l41_0, l41_1, l41_2, l41_3, l41_4, l41_5, l41_6, l41_7;
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(l41_0) : "l"(workspace + 0));
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(l41_1) : "l"(workspace + 32));
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(l41_2) : "l"(workspace + 64));
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(l41_3) : "l"(workspace + 96));
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(l41_4) : "l"(workspace + 128));
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(l41_5) : "l"(workspace + 160));
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(l41_6) : "l"(workspace + 192));
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(l41_7) : "l"(workspace + 224));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= l41_0 ^ l41_1 ^ l41_2 ^ l41_3 ^ l41_4 ^ l41_5 ^ l41_6 ^ l41_7;
        total_dt += (long long)(t1 - t0);

#elif MODE == 42  // 32 cold-DRAM loads + CCTL (heavy in-flight load)
        unsigned int base_e = ((unsigned)v * 0x9E3779B1u) & 0x0FFFFFFFu;
        int sum_e = 0;
        int ld42[32];
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(ld42[k]) : "l"(workspace + ((base_e + k*1024) & 0x0FFFFFFFu)));
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        #pragma unroll
        for (int k = 0; k < 32; k++) sum_e ^= ld42[k];
        v ^= sum_e;
        total_dt += (long long)(t1 - t0);

#elif MODE == 43  // CHAIN-DEP single DRAM load: load result feeds t0 capture (forces completion)
        // The trick: loaded value gates the BRANCH that contains the timed window.
        // If BRA is conditional on loaded value, load MUST complete before t0 reads clock.
        int loaded43;
        unsigned int addr_off43 = ((unsigned)v * 0x9E3779B1u) & 0x0FFFFFFFu;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(loaded43) : "l"(workspace + addr_off43));
        // Force chain: address of t0-clock-mov depends on loaded value (via branch on impossible cond)
        unsigned long long t0_43, t1_43;
        // Just use loaded as input register to force WAR ordering
        asm volatile("mov.u64 %0, %%clock64;\n\t"
                     "// loaded=%1\n"
                     : "=l"(t0_43) : "r"(loaded43) : "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1_43) :: "memory");
        v ^= loaded43;
        total_dt += (long long)(t1_43 - t0_43);

#elif MODE == 44  // baseline: NO load, just CCTL on idle pipeline (no nanosleep)
        // Compare to MODE 30/31/32 — if 44 also ~315 cy, then load is irrelevant
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);

#elif MODE == 50  // 1 ld.volatile (forced L2/DRAM, no CSE) + CCTL
        // ld.volatile bypasses L1, no CSE possible → forces L2/DRAM round-trip
        int loaded50;
        unsigned int addr50 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;  // 4 MB, L2-hit after 1st pass
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(loaded50) : "l"(workspace + addr50));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded50;
        total_dt += (long long)(t1 - t0);

#elif MODE == 51  // 1 ld.volatile DRAM (1 GB span, cold) + CCTL
        int loaded51;
        unsigned int addr51 = ((((unsigned)v + (unsigned)it) * 0x9E3779B1u)) & 0x0FFFFFFCu;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(loaded51) : "l"(workspace + (addr51>>2)));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded51;
        total_dt += (long long)(t1 - t0);

#elif MODE == 52  // 1 ld.global.ca with truly-warm L1 (volatile load → kills L1, then warm) + CCTL
        // First force a fence to drain prior, then warm L1 with same address (no CCTL between)
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)2000));
        // Warm L1 by ld.ca
        int warm52, timed52;
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(warm52) : "l"(workspace + 0));
        v ^= warm52;
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)2000));  // ensure warm load drained, L1 populated
        // Now timed: same address, should be L1-hit
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(timed52) : "l"(workspace + 0));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= timed52;
        total_dt += (long long)(t1 - t0);

#elif MODE == 53  // 1 LD without any modifier hint (default, ld.global) + CCTL — what does compiler emit?
        int loaded53;
        unsigned int addr53 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(loaded53) : "l"(workspace + addr53));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded53;
        total_dt += (long long)(t1 - t0);

#elif MODE == 70  // 1 ld.relaxed.gpu (L2-hit) + CCTL — does relaxed defeat the drain?
        int loaded70;
        unsigned int addr70 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(loaded70) : "l"(workspace + addr70));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded70;
        total_dt += (long long)(t1 - t0);

#elif MODE == 71  // 1 ld.relaxed.gpu (DRAM-cold) + CCTL
        int loaded71;
        unsigned int addr71 = ((((unsigned)v + (unsigned)it) * 0x9E3779B1u)) & 0x0FFFFFFCu;
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(loaded71) : "l"(workspace + (addr71>>2)));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded71;
        total_dt += (long long)(t1 - t0);

#elif MODE == 72  // 1 ld.relaxed.cta (CTA scope, weakest) + CCTL.gpu
        int loaded72;
        unsigned int addr72 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        asm volatile("ld.relaxed.cta.global.u32 %0, [%1];" : "=r"(loaded72) : "l"(workspace + addr72));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded72;
        total_dt += (long long)(t1 - t0);

#elif MODE == 73  // 1 ld.weak.global (default = weak) + CCTL — vs ld.relaxed
        int loaded73;
        unsigned int addr73 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        asm volatile("ld.weak.global.u32 %0, [%1];" : "=r"(loaded73) : "l"(workspace + addr73));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded73;
        total_dt += (long long)(t1 - t0);

#elif MODE == 100  // 1 LOAD L2-hit + release.gpu — does release drain loads?
        int loaded100;
        unsigned int addr100 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(loaded100) : "l"(workspace + addr100));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded100;
        total_dt += (long long)(t1 - t0);
#elif MODE == 101  // 1 LOAD L1-hit + release.gpu (using ld.ca, pre-warmed)
        int loaded101, warm101;
        // Pre-warm
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)1000));
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(warm101) : "l"(workspace + 0));
        v ^= warm101;
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)1000));
        // Timed
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(loaded101) : "l"(workspace + 0));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded101;
        total_dt += (long long)(t1 - t0);
#elif MODE == 102  // 1 LOAD L2-hit + fence.release.cta — does CTA-scope release drain L2 loads?
        int loaded102;
        unsigned int addr102 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(loaded102) : "l"(workspace + addr102));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.cta;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded102;
        total_dt += (long long)(t1 - t0);
#elif MODE == 103  // FFMA chain (no memory) + release.gpu — pure intrinsic release cost
        float fa103 = (float)v + (float)it * 0.5f;
        #pragma unroll 8
        for (int q = 0; q < 8; q++) fa103 = fa103 * 1.0001f + 1.0f;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : "f"(fa103) : "memory");
        asm volatile("fence.release.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= (int)fa103;
        total_dt += (long long)(t1 - t0);

#elif MODE == 90  // 1 store small-WS L2-hit + release — store-tier baseline
        unsigned int addr90 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;  // 1 MB
        int sv90 = v + (int)it;
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + addr90), "r"(sv90) : "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 91  // 1 store DRAM-cold + release — does store-tier matter for release?
        unsigned int addr91 = ((((unsigned)v + (unsigned)it) * 0x9E3779B1u)) & 0x0FFFFFFCu;
        int sv91 = v + (int)it;
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + (addr91>>2)), "r"(sv91) : "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 92  // 1 LOAD + fence.release — does release ignore loads?
        int loaded92;
        unsigned int addr92 = ((((unsigned)v + (unsigned)it) * 0x9E3779B1u)) & 0x0FFFFFFCu;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(loaded92) : "l"(workspace + (addr92>>2)));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded92;
        total_dt += (long long)(t1 - t0);
#elif MODE == 93  // 1 LOAD DRAM + 1 STORE DRAM + fence.acq_rel — should drain BOTH
        int loaded93;
        unsigned int addr93 = ((((unsigned)v + (unsigned)it) * 0x9E3779B1u)) & 0x0FFFFFFCu;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(loaded93) : "l"(workspace + (addr93>>2)));
        int sv93 = v + (int)it;
        unsigned int addr93b = (addr93 + 0x40000) & 0x0FFFFFFCu;
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + (addr93b>>2)), "r"(sv93) : "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded93;
        total_dt += (long long)(t1 - t0);
#elif MODE == 94  // 1 STORE + fence.release.cta — CTA-scope release
        unsigned int addr94 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        int sv94 = v + (int)it;
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + addr94), "r"(sv94) : "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.cta;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 95  // 1 STORE + fence.release.sys — system-scope (NVLink visibility)
        unsigned int addr95 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        int sv95 = v + (int)it;
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + addr95), "r"(sv95) : "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.sys;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 96  // 1 LOAD volatile L2 + fence.acquire.cta (CTA scope acquire — should be cheap)
        int loaded96;
        unsigned int addr96 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(loaded96) : "l"(workspace + addr96));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.cta;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded96;
        total_dt += (long long)(t1 - t0);
#elif MODE == 97  // 1 LOAD volatile L2 + fence.acquire.sys
        int loaded97;
        unsigned int addr97 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(loaded97) : "l"(workspace + addr97));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.sys;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded97;
        total_dt += (long long)(t1 - t0);
#elif MODE == 98  // atomicCAS + fence.acquire — does fence drain RMW?
        int cas_old98;
        unsigned int addr98 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        int cmp = v;
        int set = v + (int)it + 1;
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;"
                     : "=r"(cas_old98)
                     : "l"(workspace + addr98), "r"(cmp), "r"(set)
                     : "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= cas_old98;
        total_dt += (long long)(t1 - t0);

#elif MODE == 80  // 1 store + CCTL — does acquire fence drain stores?
        unsigned int addr80 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        int store_v80 = v + (int)it;
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + addr80), "r"(store_v80) : "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);

#elif MODE == 81  // 1 atomicAdd (atom.add) + CCTL — does fence drain atomic?
        unsigned int addr81 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        int atom_old81;
        asm volatile("atom.global.add.u32 %0, [%1], 1;" : "=r"(atom_old81) : "l"(workspace + addr81) : "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= atom_old81;
        total_dt += (long long)(t1 - t0);

#elif MODE == 82  // 1 store + CCTL.RELEASE (fence.release.gpu) — does release drain stores?
        unsigned int addr82 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        int store_v82 = v + (int)it;
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + addr82), "r"(store_v82) : "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);

#elif MODE == 83  // 8 stores + CCTL acquire — bulk store drain via acquire?
        unsigned int base83 = (((unsigned)v + (unsigned)it) * 4096u) & 0x0FFFF000u;
        int sv83 = v + (int)it;
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base83 + 0   *128) & 0x0FFFFFFFu)), "r"(sv83) : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base83 + 1*128) & 0x0FFFFFFFu)), "r"(sv83) : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base83 + 2*128) & 0x0FFFFFFFu)), "r"(sv83) : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base83 + 3*128) & 0x0FFFFFFFu)), "r"(sv83) : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base83 + 4*128) & 0x0FFFFFFFu)), "r"(sv83) : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base83 + 5*128) & 0x0FFFFFFFu)), "r"(sv83) : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base83 + 6*128) & 0x0FFFFFFFu)), "r"(sv83) : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base83 + 7*128) & 0x0FFFFFFFu)), "r"(sv83) : "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);

#elif MODE == 84  // 8 stores + fence.release.gpu — bulk store drain via release
        unsigned int base84 = (((unsigned)v + (unsigned)it) * 4096u) & 0x0FFFF000u;
        int sv84 = v + (int)it;
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base84 + 0*128) & 0x0FFFFFFFu)), "r"(sv84) : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base84 + 1*128) & 0x0FFFFFFFu)), "r"(sv84) : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base84 + 2*128) & 0x0FFFFFFFu)), "r"(sv84) : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base84 + 3*128) & 0x0FFFFFFFu)), "r"(sv84) : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base84 + 4*128) & 0x0FFFFFFFu)), "r"(sv84) : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base84 + 5*128) & 0x0FFFFFFFu)), "r"(sv84) : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base84 + 6*128) & 0x0FFFFFFFu)), "r"(sv84) : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + ((base84 + 7*128) & 0x0FFFFFFFu)), "r"(sv84) : "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);

#elif MODE == 110  // LDG volatile DRAM + CS2R t0 + CS2R t1 (NO FENCE) — does load delay clock?
        int loaded110;
        unsigned int addr110 = ((((unsigned)v + (unsigned)it) * 0x9E3779B1u)) & 0x0FFFFFFCu;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(loaded110) : "l"(workspace + (addr110>>2)));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        // No fence — just measure time between 2 clocks after the STRONG load
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded110;
        total_dt += (long long)(t1 - t0);
#elif MODE == 111  // LDG volatile DRAM + CS2R t0 + NOP(pad) + CS2R t1 — test load pipeline past t0
        int loaded111;
        unsigned int addr111 = ((((unsigned)v + (unsigned)it) * 0x9E3779B1u)) & 0x0FFFFFFCu;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(loaded111) : "l"(workspace + (addr111>>2)));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        // Insert IADD chain (doesn't depend on load) — if load stall is local to CS2R, these run fine
        int pad = (int)it;
        #pragma unroll 8
        for (int q = 0; q < 8; q++) pad = pad * 3 + 1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : "r"(pad) : "memory");
        v ^= loaded111;
        total_dt += (long long)(t1 - t0);
#elif MODE == 112  // LDG volatile DRAM + CONSUME(chain) + CS2R t0 + CS2R t1 — load forced complete BEFORE t0
        int loaded112;
        unsigned int addr112 = ((((unsigned)v + (unsigned)it) * 0x9E3779B1u)) & 0x0FFFFFFCu;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(loaded112) : "l"(workspace + (addr112>>2)));
        // Consume loaded value BEFORE t0 (force load to complete)
        // SHFL broadcast to force register-read consumption
        int consumed;
        asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;" : "=r"(consumed) : "r"(loaded112));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : "r"(consumed) : "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded112;
        total_dt += (long long)(t1 - t0);
#elif MODE == 113  // LDG volatile DRAM + big SHFL stall + CS2R t0 + CCTL + CS2R t1
        // Like MODE 112 but with MORE compute to guarantee load fully drained
        int loaded113;
        unsigned int addr113 = ((((unsigned)v + (unsigned)it) * 0x9E3779B1u)) & 0x0FFFFFFCu;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(loaded113) : "l"(workspace + (addr113>>2)));
        // Force long chain through loaded value
        int chain = loaded113;
        #pragma unroll 32
        for (int q = 0; q < 32; q++) chain = chain * 3 + 1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : "r"(chain) : "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded113 ^ chain;
        total_dt += (long long)(t1 - t0);

#elif MODE == 75  // ld.weak L2-hit, NO following fence — pure load latency baseline
        int loaded75;
        unsigned int addr75 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("ld.weak.global.u32 %0, [%1];" : "=r"(loaded75) : "l"(workspace + addr75));
        // Force completion via register dep on t1 — but t1 is just clock, so we need to chain
        // Use loaded75 as input to dummy register write, then read clock
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : "r"(loaded75));
        v ^= loaded75;
        total_dt += (long long)(t1 - t0);

#elif MODE == 76  // ld.weak L2-hit, then NOP-pad, then CCTL — can compiler avoid promoting?
        int loaded76;
        unsigned int addr76 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        asm volatile("ld.weak.global.u32 %0, [%1];" : "=r"(loaded76) : "l"(workspace + addr76));
        // Long FMA chain in between — does compiler still promote LDG?
        float fa = (float)loaded76;
        #pragma unroll 8
        for (int q = 0; q < 8; q++) fa = fa * 1.0001f + 1.0f;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : "f"(fa) : "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded76 ^ (int)fa;
        total_dt += (long long)(t1 - t0);

#elif MODE == 74  // 8 ld.relaxed.gpu (DRAM-cold) + CCTL — does CCTL still drain bulk?
        unsigned int base74 = ((((unsigned)v + (unsigned)it) * 0x9E3779B1u)) & 0x0FFFFF00u;
        int l74_0, l74_1, l74_2, l74_3, l74_4, l74_5, l74_6, l74_7;
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(l74_0) : "l"(workspace + ((base74 + 0   *4096) & 0x0FFFFFFFu)));
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(l74_1) : "l"(workspace + ((base74 + 1   *4096) & 0x0FFFFFFFu)));
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(l74_2) : "l"(workspace + ((base74 + 2   *4096) & 0x0FFFFFFFu)));
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(l74_3) : "l"(workspace + ((base74 + 3   *4096) & 0x0FFFFFFFu)));
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(l74_4) : "l"(workspace + ((base74 + 4   *4096) & 0x0FFFFFFFu)));
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(l74_5) : "l"(workspace + ((base74 + 5   *4096) & 0x0FFFFFFFu)));
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(l74_6) : "l"(workspace + ((base74 + 6   *4096) & 0x0FFFFFFFu)));
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(l74_7) : "l"(workspace + ((base74 + 7   *4096) & 0x0FFFFFFFu)));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= l74_0 ^ l74_1 ^ l74_2 ^ l74_3 ^ l74_4 ^ l74_5 ^ l74_6 ^ l74_7;
        total_dt += (long long)(t1 - t0);

#elif MODE == 60  // CHAIN-DEP: load, USE the loaded value to compute a noop on t0 → forces serial completion
        // Pattern: addr2 = addr ^ (loaded & 0); time the CCTL after load is provably done
        int loaded60;
        unsigned int addr60 = (((unsigned)v + (unsigned)it) * 64u) & 0x000FFFFFu;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(loaded60) : "l"(workspace + addr60));
        // Use loaded60 in a way that produces 0 but forces the consumer to wait for the load
        // SHFL.IDX with a value-derived lane index forces register read with completed value
        unsigned int dummy_idx;
        asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;" : "=r"(dummy_idx) : "r"(loaded60));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= loaded60 ^ (int)dummy_idx;
        total_dt += (long long)(t1 - t0);
#elif MODE == 20  // fence.sc.gpu (sequential consistency, strongest)
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.sc.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 21  // fence.sc.cta
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.sc.cta;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 22  // fence.sc.sys
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.sc.sys;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 23  // fence.acq_rel.cta
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acq_rel.cta;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 24  // chained MEMBAR.ALL.GPU — does it saturate like CCTL chain?
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        #pragma unroll N_CHAIN_MEMBAR
        for (int j = 0; j < N_CHAIN_MEMBAR; j++) {
            asm volatile("fence.release.gpu;" ::: "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 14  // MEMBAR.ALL.GPU after writes WITHOUT drain — measure release "drain time"
        for (int j = 0; j < WS_INTS; j += 32) {
            int rval = v + j;
            asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + j), "r"(rval) : "memory");
        }
        // NO nanosleep here — measures "writes still in flight + MEMBAR drain"
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 5  // WRITES with drain (st.global.wb after nanosleep)
        for (int j = 0; j < WS_INTS; j += 32) {
            int rval = v + j;
            asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + j), "r"(rval) : "memory");
        }
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 4
        // Same as MODE 3 but timed measurement is the SECOND CCTL (L1 already empty)
        int fill_v = v;
        for (int j = 0; j < WS_INTS; j += 32) {
            int loaded;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(loaded) : "l"(workspace + j));
            fill_v ^= loaded;
        }
        v ^= fill_v;
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("fence.acquire.gpu;" ::: "memory");  // first invalidates L1
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)1000));  // tiny gap
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");  // second on already-empty L1
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#endif
    }

    // Anti-DCE
    C[blockIdx.x + 32] = (float)v;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N_FILL_KB=%d SLEEP_NS=%u N_OUTER=%d cy/iter=%.2f\n",
               MODE, N_FILL_KB, (unsigned)SLEEP_NS, N_OUTER, (double)total_dt/N_OUTER);
    }
}
