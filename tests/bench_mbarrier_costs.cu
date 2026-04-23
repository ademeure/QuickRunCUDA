// Measure mbarrier op costs directly (no hot-loop, single-thread, no DCE risk).

#ifndef MODE
// 0 = mbarrier.arrive (just the arrive)
// 1 = mbarrier.arrive + mbarrier.try_wait (wait until self-arrived)
// 2 = mbarrier.init only
// 3 = mbarrier.test_wait (always returns immediately, just opcode cost)
// 4 = mbarrier.inval
// 5 = full pattern: init + arrive + try_wait + (re-init for next iter)
#define MODE 0
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    __shared__ __align__(8) unsigned long long mb;

    unsigned int mb_addr = __cvta_generic_to_shared(&mb);

    // Init
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mb_addr));

    unsigned long long t0 = 0, t1 = 0;
    long long total_dt = 0;
    unsigned long long state = 0;

    for (int it = 0; it < N_OUTER; it++) {
#if MODE == 0
        // arrive only — re-init each iter to keep barrier fresh
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mb_addr));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(mb_addr));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

#elif MODE == 1
        // arrive then try_wait until satisfied (self-completion)
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mb_addr));
        asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(mb_addr));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        int done = 0;
        while (done == 0) {
            int pred;
            asm volatile("{ .reg .pred %%p;\n"
                         "  mbarrier.try_wait.shared.b64 %%p, [%1], %2;\n"
                         "  selp.b32 %0, 1, 0, %%p; }"
                         : "=r"(pred) : "r"(mb_addr), "l"(state));
            done = pred;
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

#elif MODE == 2
        // init only
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mb_addr));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

#elif MODE == 3
        // test_wait (non-blocking) on satisfied barrier
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mb_addr));
        asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(mb_addr));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        int pred;
        asm volatile("{ .reg .pred %%p;\n"
                     "  mbarrier.test_wait.shared.b64 %%p, [%1], %2;\n"
                     "  selp.b32 %0, 1, 0, %%p; }"
                     : "=r"(pred) : "r"(mb_addr), "l"(state));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

#elif MODE == 4
        // inval
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mb_addr));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(mb_addr));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

#elif MODE == 5
        // Full round-trip: init + arrive + try_wait
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mb_addr));
        asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(mb_addr));
        int done2 = 0;
        while (done2 == 0) {
            int pred;
            asm volatile("{ .reg .pred %%p;\n"
                         "  mbarrier.try_wait.shared.b64 %%p, [%1], %2;\n"
                         "  selp.b32 %0, 1, 0, %%p; }"
                         : "=r"(pred) : "r"(mb_addr), "l"(state));
            done2 = pred;
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

#elif MODE == 6
        // bar.sync (a.k.a. __syncthreads with single warp)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("bar.sync 0;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

#elif MODE == 7
        // bar.arrive + bar.sync_count (Hopper+ split barrier)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("bar.arrive 1, 32;");
        asm volatile("bar.sync 1, 32;");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
#endif

        total_dt += (long long)(t1 - t0);
    }

    ((unsigned long long*)C)[1024] = (unsigned long long)total_dt;
    // Anti-DCE
    if (state == 0xDEADBEEFDEADBEEFull) C[0] = (float)state;
    if (mb == 0xDEADBEEFDEADBEEFull) C[1] = (float)mb;
}
