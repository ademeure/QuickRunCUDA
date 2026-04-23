// Force 2 CTAs on the SAME SM via high occupancy.
// Launch many CTAs; the FIRST two CTAs that land on the same SM act as the test pair.
//
// We launch enough CTAs that multiple share an SM. CTAs that land first on each SM
// register themselves as "loader" (warp 0 issues many DRAM loads).
// CTAs that land second on the same SM act as "fencer" (warp 1 nanosleeps + fences).
//
// 64 threads/CTA × 8 CTAs/SM = 512 threads/SM (well below max 2048).
// Launch 2*148 = 296 CTAs to ensure same-SM pairing.

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
#define FENCE_MODE 1  // 0=acquire.gpu, 1=release.gpu, 2=sc.gpu, 3=acquire.cta, 4=release.cta
#endif

extern "C" __global__ __launch_bounds__(64, 8)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid & 31;
    int bid = blockIdx.x;

    int* workspace = (int*)A;
    // SM-paired counters in the second-to-last region of A
    // Each SM has 2 slots: slot 0 = loader_bid, slot 1 = fencer_bid
    unsigned int* sm_pair = (unsigned int*)(A + (256 * 1024 * 1024 - 4096));

    // Discover which SM we're on
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

    // Race to claim "loader" or "fencer" slot for this SM
    int role = -1;  // -1 = idle, 0 = loader, 1 = fencer
    if (warp == 0 && lane == 0) {
        unsigned int prev = atomicCAS(&sm_pair[smid * 2], 0u, (unsigned)bid + 1);
        if (prev == 0u) {
            role = 0;  // I'm the loader
        } else {
            // Try fencer slot
            unsigned int prev2 = atomicCAS(&sm_pair[smid * 2 + 1], 0u, (unsigned)bid + 1);
            if (prev2 == 0u) {
                role = 1;  // I'm the fencer
            }
        }
    }
    // Broadcast role to all threads
    role = __shfl_sync(0xffffffff, role, 0);

    if (role == -1) return;  // idle, this CTA doesn't participate

    unsigned long long t0 = 0, t1 = 0;
    long long total_dt = 0;
    int v = lane + 1;

    for (int it = 0; it < N_OUTER; it++) {
        if (role == 0) {
            // Loader: continuously issue DRAM-cold loads
            int accum = 0;
            unsigned int base = ((unsigned)(v + it + smid * 1337) * 0x9E3779B1u) & 0x0FFFFF00u;
            #pragma unroll
            for (int k = 0; k < N_LOADS; k++) {
                int x;
                unsigned int addr = (base + k * 4096) & 0x0FFFFFFFu;
                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(x) : "l"(workspace + (addr >> 2)));
                accum ^= x;
            }
            if (accum == 0xDEADBEEF) C[bid] = (float)accum;
        } else if (role == 1) {
            // Fencer
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

    // Each fencer writes its smid + total cycles
    if (role == 1 && warp == 0 && lane == 0) {
        unsigned int slot = smid;  // up to 148 fencers possible
        ((unsigned long long*)C)[1024 + slot * 2] = (unsigned long long)total_dt;
        ((unsigned long long*)C)[1024 + slot * 2 + 1] = (unsigned long long)smid;  // for verification
    }
}
