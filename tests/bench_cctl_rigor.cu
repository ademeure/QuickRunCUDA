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
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

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
