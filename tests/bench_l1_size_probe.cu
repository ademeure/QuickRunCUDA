// Probe effective L1 size by sweeping working set size and finding the latency cliff.
// Also takes dynamic smem to force a specific carveout.
//
// -H "#define WS_KB <n>"  -- working set in KB (test fills + re-reads this much)
// -s <bytes>              -- dynamic smem (forces L1+SMEM to allocate at least this much smem)

#ifndef WS_KB
#define WS_KB 4
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif

extern __shared__ int dyn_smem[];

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int* workspace = (int*)A;
    const int N_INTS = (WS_KB * 1024) / 4;  // total ints in WS
    const int N_LINES = N_INTS / 32;        // 128B cache lines

    // Fill (warm L1 if it fits)
    int fill_v = 0;
    for (int k = 0; k < N_LINES; k++) {
        int x;
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(workspace + k * 32));
        fill_v ^= x;
    }
    // Note: dyn_smem only used if -s > 0 to force carveout; never read inside kernel

    long long total_dt = 0;
    int v = fill_v;

    for (int it = 0; it < N_OUTER; it++) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        int s = 0;
        for (int k = 0; k < N_LINES; k++) {
            int x;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(workspace + k * 32));
            s ^= x;
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= s;
        total_dt += (long long)(t1 - t0);
    }

    ((unsigned long long*)C)[1024] = (unsigned long long)total_dt;
    if (v == 0xCAFEBABE) C[0] = (float)v;
}
