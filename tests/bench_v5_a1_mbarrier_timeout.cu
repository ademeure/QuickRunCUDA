// V5 A1: mbarrier.try_wait suspendTimeHint test
// Test if the timeout hint actually changes wait behavior.
// Pattern: thread 0 holds for HOLD_NS, others wait via mbarrier.try_wait with HINT_NS hint.
// Measure: cycles spent in the wait loop, retry count.
//
// HINT modes:
//   MODE 0: hint = 0 (no suspension hint)
//   MODE 1: hint = 100 (100 ns)
//   MODE 2: hint = 10000 (10 µs — should be > HOLD)
//   MODE 3: hint = 1000000 (1 ms — way more than needed)
//   MODE 4: hint = 0xFFFFFFFF (max u32 — does it cap?)
#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
#define HINT 0u
#elif MODE == 1
#define HINT 100u
#elif MODE == 2
#define HINT 10000u
#elif MODE == 3
#define HINT 1000000u
#elif MODE == 4
#define HINT 0xFFFFFFFFu
#endif

extern "C" __global__ __launch_bounds__(128, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bar;
    unsigned int bar_addr = __cvta_generic_to_shared(&bar);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                     :: "r"(bar_addr), "r"((unsigned int)128));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Thread 0: hold for HOLD ns then arrive
    // Others: arrive then try_wait with HINT timeout in a loop, count retries
    if (threadIdx.x == 0) {
        // Hold for ITERS µs
        for (int i = 0; i < ITERS; i++) {
            __nanosleep(1000);
        }
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                     :: "r"(bar_addr));
    } else {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                     :: "r"(bar_addr));

        // Loop with try_wait + HINT, count retries
        unsigned int retries = 0;
        unsigned int cur_hint = HINT;
        while (1) {
            unsigned int p;
            asm volatile("{ .reg .pred q;\n"
                         "  mbarrier.try_wait.parity.shared::cta.b64 q, [%1], 0, %2;\n"
                         "  selp.u32 %0, 1, 0, q; }"
                         : "=r"(p) : "r"(bar_addr), "r"(cur_hint));
            if (p) break;
            retries++;
            if (retries > 1000000) break;  // safety cap
        }
        if (threadIdx.x == 1) {
            // Print retries from a single non-trigger thread
            printf("MODE=%d HINT=%u retries=%u\n", MODE, HINT, retries);
        }
    }

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        printf("MODE=%d HINT=%u total_cy=%llu\n", MODE, HINT, t1-t0);
    }
}
