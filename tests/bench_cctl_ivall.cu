// CCTL.IVALL characterization on B300 sm_103a (1800-locked).
// PTX `fence.acquire.gpu` compiles to SASS `CCTL.IVALL` on this rig.
// MODES (set via -H "#define MODE N"):
//   0 = baseline empty (clock64 noise floor)
//   1 = single CCTL.IVALL
//   2 = N_CHAIN consecutive CCTL.IVALLs (chained — cy/op when L1 already invalid)
//   3 = fill L1 with N_FILL_KB of CACHED LOADS, then 1 CCTL.IVALL
//   4 = fill L1 with N_FILL_KB of WRITES (dirty lines), then 1 CCTL.IVALL
//   5 = MODE 3 + a SECOND timed CCTL (L1 should be empty for the 2nd)
#ifndef MODE
#define MODE 1
#endif
#ifndef N_CHAIN
#define N_CHAIN 1024
#endif
#ifndef N_FILL_KB
#define N_FILL_KB 16
#endif
#ifndef N_OUTER
#define N_OUTER 50
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int* workspace = (int*)A;
    const int WS_INTS = N_FILL_KB * 256;  // N_FILL_KB KB / 4 B per int
    int v = (int)threadIdx.x + 1;

    unsigned long long t0 = 0, t1 = 0;
    long long total_dt = 0;

    asm volatile("fence.acquire.gpu;" ::: "memory");

    for (int it = 0; it < N_OUTER; it++) {
#if MODE == 0
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 1
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 2
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        #pragma unroll N_CHAIN
        for (int j = 0; j < N_CHAIN; j++) {
            asm volatile("fence.acquire.gpu;" ::: "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 3
        int fill_v = v;
        for (int j = 0; j < WS_INTS; j += 32) {
            int loaded;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(loaded) : "l"(workspace + j));
            fill_v ^= loaded;
        }
        v ^= fill_v;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 4
        for (int j = 0; j < WS_INTS; j += 32) {
            asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + j), "r"(v + j) : "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 6  // st.global.cg (L2-only, NO L1 — baseline for "no dirty L1")
        for (int j = 0; j < WS_INTS; j += 32) {
            int rval = v + j;
            asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(workspace + j), "r"(rval) : "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 7  // plain st.global (no hint, compiler default)
        for (int j = 0; j < WS_INTS; j += 32) {
            int rval = v + j;
            asm volatile("st.global.u32 [%0], %1;" :: "l"(workspace + j), "r"(rval) : "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 8  // load THEN write same lines (RMW — load.ca + store.wb)
        for (int j = 0; j < WS_INTS; j += 32) {
            int loaded;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(loaded) : "l"(workspace + j));
            int rval = loaded ^ v;
            asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + j), "r"(rval) : "memory");
            v ^= loaded;
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#elif MODE == 5
        int fill_v = v;
        for (int j = 0; j < WS_INTS; j += 32) {
            int loaded;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(loaded) : "l"(workspace + j));
            fill_v ^= loaded;
        }
        v ^= fill_v;
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
#endif
    }

    // Anti-DCE: ALWAYS store v (depends on fill_v / loaded data so compiler can't elide)
    C[blockIdx.x + 32] = (float)v;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
#if MODE == 2
        double per_op = (double)total_dt / (double)N_OUTER / (double)N_CHAIN;
        printf("MODE=%d N_CHAIN=%d N_OUTER=%d cy/iter=%.2f cy/CCTL=%.3f\n",
               MODE, N_CHAIN, N_OUTER, (double)total_dt/N_OUTER, per_op);
#elif MODE == 3 || MODE == 4 || MODE == 5
        const char* desc = MODE==3?"loaded":MODE==4?"written":"loaded-then-2ndCCTL";
        printf("MODE=%d N_FILL_KB=%d N_OUTER=%d cy/iter=%.2f (1 CCTL after %d KB %s)\n",
               MODE, N_FILL_KB, N_OUTER, (double)total_dt/N_OUTER, N_FILL_KB, desc);
#else
        printf("MODE=%d N_OUTER=%d cy/iter=%.2f\n",
               MODE, N_OUTER, (double)total_dt/N_OUTER);
#endif
    }
}
