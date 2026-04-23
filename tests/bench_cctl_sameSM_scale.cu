// Scale test: does release.gpu drain scale with number of LOADING CTAs on same SM?
// Launch 296 CTAs at 8/SM occupancy. Per SM, claim N_LOADERS as loaders, 1 as fencer.
//
// If drain scales linearly with #loaders → SM-wide drain is "wait for all".
// If drain is constant → drain bounded by latency of slowest single load.

#ifndef N_LOADERS
#define N_LOADERS 1   // 1, 2, 4, 8 — number of loader CTAs per SM
#endif
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
#define FENCE_MODE 1
#endif

extern "C" __global__ __launch_bounds__(64, 8)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid & 31;
    int bid = blockIdx.x;

    int* workspace = (int*)A;
    // Per-SM counters: 16 slots per SM (allows up to N_LOADERS+1 roles tracked)
    unsigned int* sm_pair = (unsigned int*)(A + (256 * 1024 * 1024 - 4096));

    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

    // Race for roles: slots 0..N_LOADERS-1 are loader slots, slot N_LOADERS is fencer
    int role = -1;
    if (warp == 0 && lane == 0) {
        // Try loader slots first
        for (int s = 0; s < N_LOADERS; s++) {
            unsigned int prev = atomicCAS(&sm_pair[smid * 16 + s], 0u, (unsigned)bid + 1);
            if (prev == 0u) { role = 0; break; }
        }
        if (role == -1) {
            // Try fencer slot
            unsigned int prev = atomicCAS(&sm_pair[smid * 16 + N_LOADERS], 0u, (unsigned)bid + 1);
            if (prev == 0u) role = 1;
        }
    }
    role = __shfl_sync(0xffffffff, role, 0);

    if (role == -1) return;

    unsigned long long t0 = 0, t1 = 0;
    long long total_dt = 0;
    int v = lane + 1;

    for (int it = 0; it < N_OUTER; it++) {
        if (role == 0) {
            int accum = 0;
            unsigned int base = ((unsigned)(v + it + smid * 1337 + bid * 7) * 0x9E3779B1u) & 0x0FFFFF00u;
            #pragma unroll
            for (int k = 0; k < N_LOADS; k++) {
                int x;
                unsigned int addr = (base + k * 4096) & 0x0FFFFFFFu;
                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(x) : "l"(workspace + (addr >> 2)));
                accum ^= x;
            }
            if (accum == 0xDEADBEEF) C[bid] = (float)accum;
        } else {
            asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            #if FENCE_MODE == 0
                asm volatile("fence.acquire.gpu;" ::: "memory");
            #elif FENCE_MODE == 1
                asm volatile("fence.release.gpu;" ::: "memory");
            #elif FENCE_MODE == 2
                asm volatile("fence.sc.gpu;" ::: "memory");
            #endif
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
            total_dt += (long long)(t1 - t0);
        }
    }

    if (role == 1 && warp == 0 && lane == 0) {
        ((unsigned long long*)C)[1024 + smid * 2] = (unsigned long long)total_dt;
        ((unsigned long long*)C)[1024 + smid * 2 + 1] = (unsigned long long)smid;
    }
}
