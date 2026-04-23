// Cross-warp CCTL drain test:
// - Warp 0: launches a bunch of in-flight DRAM loads, then waits on them.
// - Warp 1: waits nanosleep N ns (so warp 0's loads are partially completed),
//   then times a fence.acquire.gpu.
// If fence drain is SM-scope: warp 1's fence cost depends on SLEEP_NS
//   (short sleep → warp 0's loads still in flight → fence must drain them).
// If fence drain is per-warp: warp 1's fence is cheap regardless.

#ifndef SLEEP_NS
#define SLEEP_NS 10
#endif
#ifndef N_LOADS
#define N_LOADS 32  // how many loads warp 0 issues
#endif
#ifndef N_OUTER
#define N_OUTER 50
#endif
#ifndef MODE
#define MODE 0  // 0 = cross-warp test, 1 = no-warp0-activity baseline, 2 = warp 0 also fences (SC fence sync)
#endif

extern "C" __global__ __launch_bounds__(64, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (blockIdx.x != 0) return;
    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid & 31;

    int* workspace = (int*)A;
    unsigned long long t0 = 0, t1 = 0;
    long long total_dt = 0;
    int v = lane + 1;

    for (int it = 0; it < N_OUTER; it++) {
#if MODE == 0
        // Cross-warp test
        if (warp == 0) {
            // Issue N_LOADS DRAM-cold volatile loads
            int accum = 0;
            unsigned int base = ((unsigned)(v + it) * 0x9E3779B1u) & 0x0FFFFF00u;
            #pragma unroll
            for (int k = 0; k < N_LOADS; k++) {
                int x;
                unsigned int addr = (base + k * 4096) & 0x0FFFFFFFu;
                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(x) : "l"(workspace + (addr >> 2)));
                accum ^= x;
            }
            // Consume to prevent DCE
            if (accum == 0xDEADBEEF) C[lane] = (float)accum;
        } else {
            // Warp 1: short nanosleep, then time fence
            asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            asm volatile("fence.acquire.gpu;" ::: "memory");
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
            total_dt += (long long)(t1 - t0);
        }
#elif MODE == 1
        // Baseline: warp 0 does NOTHING, warp 1 times fence (should be ~3 cy)
        if (warp == 1) {
            asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            asm volatile("fence.acquire.gpu;" ::: "memory");
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
            total_dt += (long long)(t1 - t0);
        }
#elif MODE == 2
        // Warp 0 loads, Warp 1 also loads (same behavior) + fence
        if (warp == 0) {
            int accum = 0;
            unsigned int base = ((unsigned)(v + it) * 0x9E3779B1u) & 0x0FFFFF00u;
            #pragma unroll
            for (int k = 0; k < N_LOADS; k++) {
                int x;
                unsigned int addr = (base + k * 4096) & 0x0FFFFFFFu;
                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(x) : "l"(workspace + (addr >> 2)));
                accum ^= x;
            }
            if (accum == 0xDEADBEEF) C[lane] = (float)accum;
        } else {
            asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
            // Warp 1 also issues 1 DRAM load
            int x;
            unsigned int addr = ((unsigned)(v + it) * 0x123456u) & 0x0FFFFFFFu;
            asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(x) : "l"(workspace + (addr >> 2)));
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            asm volatile("fence.acquire.gpu;" ::: "memory");
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
            if (x == 0xDEADBEEF) C[lane+1] = (float)x;
            total_dt += (long long)(t1 - t0);
        }
#elif MODE == 3
        // Warp 0 loads, Warp 1 times fence.release.gpu (MEMBAR.ALL.GPU)
        if (warp == 0) {
            int accum = 0;
            unsigned int base = ((unsigned)(v + it) * 0x9E3779B1u) & 0x0FFFFF00u;
            #pragma unroll
            for (int k = 0; k < N_LOADS; k++) {
                int x;
                unsigned int addr = (base + k * 4096) & 0x0FFFFFFFu;
                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(x) : "l"(workspace + (addr >> 2)));
                accum ^= x;
            }
            if (accum == 0xDEADBEEF) C[lane] = (float)accum;
        } else {
            asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            asm volatile("fence.release.gpu;" ::: "memory");
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
            total_dt += (long long)(t1 - t0);
        }
#elif MODE == 4
        // Warp 0 STORES, Warp 1 times fence.release.gpu — does release drain cross-warp stores?
        if (warp == 0) {
            unsigned int base = ((unsigned)(v + it) * 0x9E3779B1u) & 0x0FFFFF00u;
            #pragma unroll
            for (int k = 0; k < N_LOADS; k++) {
                unsigned int addr = (base + k * 4096) & 0x0FFFFFFFu;
                int val = (int)(v + k + it);
                asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + (addr >> 2)), "r"(val) : "memory");
            }
        } else {
            asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            asm volatile("fence.release.gpu;" ::: "memory");
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
            total_dt += (long long)(t1 - t0);
        }
#elif MODE == 5
        // Warp 0 loads, Warp 1 times fence.sc.gpu (full GPU-scope sequential consistency)
        if (warp == 0) {
            int accum = 0;
            unsigned int base = ((unsigned)(v + it) * 0x9E3779B1u) & 0x0FFFFF00u;
            #pragma unroll
            for (int k = 0; k < N_LOADS; k++) {
                int x;
                unsigned int addr = (base + k * 4096) & 0x0FFFFFFFu;
                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(x) : "l"(workspace + (addr >> 2)));
                accum ^= x;
            }
            if (accum == 0xDEADBEEF) C[lane] = (float)accum;
        } else {
            asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            asm volatile("fence.sc.gpu;" ::: "memory");
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
            total_dt += (long long)(t1 - t0);
        }
#elif MODE == 7
        // Warp 0 LOADS, Warp 1 times fence.release.cta (CTA-scope release)
        if (warp == 0) {
            int accum = 0;
            unsigned int base = ((unsigned)(v + it) * 0x9E3779B1u) & 0x0FFFFF00u;
            #pragma unroll
            for (int k = 0; k < N_LOADS; k++) {
                int x;
                unsigned int addr = (base + k * 4096) & 0x0FFFFFFFu;
                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(x) : "l"(workspace + (addr >> 2)));
                accum ^= x;
            }
            if (accum == 0xDEADBEEF) C[lane] = (float)accum;
        } else {
            asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            asm volatile("fence.release.cta;" ::: "memory");
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
            total_dt += (long long)(t1 - t0);
        }
#elif MODE == 8
        // Warp 0 STORES, Warp 1 times fence.release.cta — does cta-scope drain stores cross-warp?
        if (warp == 0) {
            unsigned int base = ((unsigned)(v + it) * 0x9E3779B1u) & 0x0FFFFF00u;
            #pragma unroll
            for (int k = 0; k < N_LOADS; k++) {
                unsigned int addr = (base + k * 4096) & 0x0FFFFFFFu;
                int val = (int)(v + k + it);
                asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + (addr >> 2)), "r"(val) : "memory");
            }
        } else {
            asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            asm volatile("fence.release.cta;" ::: "memory");
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
            total_dt += (long long)(t1 - t0);
        }
#elif MODE == 6
        // Warp 0 STORES, Warp 1 times fence.sc.sys (NVLink-scope)
        if (warp == 0) {
            unsigned int base = ((unsigned)(v + it) * 0x9E3779B1u) & 0x0FFFFF00u;
            #pragma unroll
            for (int k = 0; k < N_LOADS; k++) {
                unsigned int addr = (base + k * 4096) & 0x0FFFFFFFu;
                int val = (int)(v + k + it);
                asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + (addr >> 2)), "r"(val) : "memory");
            }
        } else {
            asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            asm volatile("fence.release.sys;" ::: "memory");
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
            total_dt += (long long)(t1 - t0);
        }
#endif
    }

    if (warp == 1 && lane == 0) {
        ((unsigned long long*)C)[1024] = (unsigned long long)total_dt;
    }
}
