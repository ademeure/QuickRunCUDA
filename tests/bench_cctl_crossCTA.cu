// Cross-CTA fence drain scope test:
// CTA 0 issues N_LOADS DRAM-cold volatile loads continuously.
// CTA 1 does short nanosleep then times a fence (acquire or release).
// If fence scope is CTA-local: CTA 1's fence cost should NOT depend on CTA 0's activity.
// If fence scope is SM-wide: CTA 1's fence cost depends on whether CTAs 0&1 share an SM.
// If fence scope is GPU-wide: CTA 1's fence waits for ALL CTAs' loads (everywhere on GPU).
//
// With 2 CTAs × 64 threads: both may be on same SM or different SMs (scheduler-dependent).
// To force co-residence: use cluster launch or persistent threads.

#ifndef SLEEP_NS
#define SLEEP_NS 0
#endif
#ifndef N_LOADS
#define N_LOADS 32
#endif
#ifndef N_OUTER
#define N_OUTER 50
#endif
#ifndef FENCE_MODE
// 0 = acquire.gpu, 1 = release.gpu, 2 = sc.gpu
#define FENCE_MODE 1
#endif
#ifndef CTA0_ACTION
// 0 = loads, 1 = stores
#define CTA0_ACTION 0
#endif

extern "C" __global__ __launch_bounds__(64, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid / 32;

    int* workspace = (int*)A;
    unsigned long long t0 = 0, t1 = 0;
    long long total_dt = 0;
    int v = lane + 1;

    for (int it = 0; it < N_OUTER; it++) {
        if (bid == 0) {
            // CTA 0: issue many in-flight memory ops
            unsigned int base = ((unsigned)(v + it + bid*1337) * 0x9E3779B1u) & 0x0FFFFF00u;
            #if CTA0_ACTION == 0
                int accum = 0;
                #pragma unroll
                for (int k = 0; k < N_LOADS; k++) {
                    int x;
                    unsigned int addr = (base + k * 4096) & 0x0FFFFFFFu;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(x) : "l"(workspace + (addr >> 2)));
                    accum ^= x;
                }
                if (accum == 0xDEADBEEF) C[lane] = (float)accum;
            #elif CTA0_ACTION == 1
                #pragma unroll
                for (int k = 0; k < N_LOADS; k++) {
                    unsigned int addr = (base + k * 4096) & 0x0FFFFFFFu;
                    int val = (int)(v + k + it);
                    asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + (addr >> 2)), "r"(val) : "memory");
                }
            #endif
        } else if (bid == 1) {
            // CTA 1: nanosleep, then time fence
            asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            #if FENCE_MODE == 0
                asm volatile("fence.acquire.gpu;" ::: "memory");
            #elif FENCE_MODE == 1
                asm volatile("fence.release.gpu;" ::: "memory");
            #elif FENCE_MODE == 2
                asm volatile("fence.sc.gpu;" ::: "memory");
            #elif FENCE_MODE == 3
                asm volatile("fence.acquire.cta;" ::: "memory");
            #elif FENCE_MODE == 4
                asm volatile("fence.release.cta;" ::: "memory");
            #endif
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
            total_dt += (long long)(t1 - t0);
        }
    }

    if (bid == 1 && warp == 0 && lane == 0) {
        ((unsigned long long*)C)[1024] = (unsigned long long)total_dt;
    }
    // Record SM ID per CTA
    if (warp == 0 && lane == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        ((unsigned int*)C)[2050 + bid] = smid;
    }
}
