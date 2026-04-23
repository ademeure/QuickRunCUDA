// Question: when fence.acquire.gpu (CCTL.IVALL) executes in CTA X,
// does it invalidate L1 only for CTA X, or for ALL CTAs co-resident on the SM?
//
// Test: 2 CTAs on SAME SM (forced via __launch_bounds__ + atomicCAS pairing).
// CTA A "warmer": issues many cached loads (ld.global.ca) → fills L1
// CTA B "invalidator": issues fence.acquire.gpu (CCTL.IVALL)
// CTA A "tester": re-reads SAME addresses, measures latency
//   - If L1-hit (~25-30 cy) → CCTL did NOT touch CTA A's L1
//   - If L2-hit (~80-100 cy) or higher → CCTL invalidated SM-wide
//
// Coordination via shared global flags. Also measure baseline (no CCTL between fill and re-read).

#ifndef N_LINES
#define N_LINES 32          // number of L1 cache lines to fill
#endif
#ifndef N_OUTER
#define N_OUTER 50
#endif
#ifndef MODE
// 0 = baseline: warmer fills L1, then re-reads (no other CTA does CCTL)
// 1 = warmer fills L1, OTHER CTA on SAME SM does CCTL.IVALL, warmer re-reads
// 2 = warmer fills L1, warmer ITSELF does CCTL.IVALL, re-reads (control)
// 3 = warmer fills L1, OTHER CTA on DIFFERENT SM does CCTL, warmer re-reads
#define MODE 0
#endif

extern __shared__ int dyn_smem[];
extern "C" __global__ __launch_bounds__(64, 8)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid & 31;
    int bid = blockIdx.x;

    int* workspace = (int*)A;
    // A is int*, so int-index. 64M ints = 256MB. Reserve last 2048 ints for control:
    //   A[64M-2048 .. 64M-1024]: sm_pair (148 SMs × 4 slots × 4 bytes = needs 592 ints)
    //   A[64M-1024 .. 64M]: handshake flags (148 SMs × 8 ints)
    unsigned int* sm_pair = (unsigned int*)(workspace + 67108864 - 2048);
    volatile unsigned int* flags = (volatile unsigned int*)(workspace + 67108864 - 1024);

    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

    // Role election: 0 = warmer, 1 = invalidator
    int role = -1;
    if (warp == 0 && lane == 0) {
#if MODE == 3
        // Want warmer + invalidator on DIFFERENT SMs.
        // Use bid even/odd as natural assignment.
        // For this mode, just use bid==0/1 directly without pairing
        if (bid == 0) role = 0;
        else if (bid == 1) role = 1;
#else
        // Same-SM: race per SM
        unsigned int prev = atomicCAS(&sm_pair[smid * 4], 0u, (unsigned)bid + 1);
        if (prev == 0u) {
            role = 0;
        } else {
            unsigned int prev2 = atomicCAS(&sm_pair[smid * 4 + 1], 0u, (unsigned)bid + 1);
            if (prev2 == 0u) role = 1;
        }
#endif
    }
    role = __shfl_sync(0xffffffff, role, 0);
    if (role == -1) return;

    if (warp != 0) return;  // only warp 0 of each role does work

    // Per-SM addresses: workspace + smid*0x10000 + line*128
    // (each cache line is 128 B = 32 ints)
    int* my_lines = workspace + (smid * 0x10000 / 4);

    long long total_dt = 0;
    int v = lane + 1;

    // Per-SM flag at flags[smid*8..]
    volatile unsigned int* my_flag = &flags[smid * 8];

    if (lane == 0 && role == 0) my_flag[0] = 0;
    __syncwarp();

    for (int it = 0; it < N_OUTER; it++) {
        if (role == 0) {
            // Warmer: fill L1 with ld.global.ca
            int sum_fill = 0;
            #pragma unroll
            for (int k = 0; k < N_LINES; k++) {
                int x;
                asm volatile("ld.global.ca.u32 %0, [%1];"
                             : "=r"(x) : "l"(my_lines + k * 32 + lane));
                sum_fill ^= x;
            }
            if (sum_fill == 0xDEADBEEF) C[bid] = (float)sum_fill;

            // Signal invalidator (only in MODES 1, 3)
#if MODE == 1 || MODE == 3
            if (lane == 0) {
                __threadfence_block();
                my_flag[0] = it + 1;
                // Wait for invalidator to finish
                while (my_flag[1] != it + 1) {}
            }
            __syncwarp();
#elif MODE == 2
            // Self-invalidate
            asm volatile("fence.acquire.gpu;" ::: "memory");
#endif

            // Re-read same lines, measure latency
            unsigned long long t0, t1;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            int sum_test = 0;
            #pragma unroll
            for (int k = 0; k < N_LINES; k++) {
                int x;
                asm volatile("ld.global.ca.u32 %0, [%1];"
                             : "=r"(x) : "l"(my_lines + k * 32 + lane));
                sum_test ^= x;
            }
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
            v ^= sum_test;
            total_dt += (long long)(t1 - t0);
        } else if (role == 1) {
#if MODE == 1 || MODE == 3
            // Wait for warmer to fill L1
            if (lane == 0) {
                while (my_flag[0] != it + 1) {}
                // Issue CCTL.IVALL
                asm volatile("fence.acquire.gpu;" ::: "memory");
                __threadfence_block();
                my_flag[1] = it + 1;
            }
            __syncwarp();
#endif
        }
    }

    if (role == 0 && lane == 0) {
        // Each warmer reports per-SM
        ((unsigned long long*)C)[1024 + smid] = (unsigned long long)total_dt;
    }
    if (v == 0xCAFEBABE) C[1] = (float)v;  // anti-DCE
}
