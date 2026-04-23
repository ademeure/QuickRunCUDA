#ifndef N_LOADERS
#define N_LOADERS 1
#endif
#ifndef SLEEP_NS
#define SLEEP_NS 0
#endif
#ifndef N_OPS
#define N_OPS 32
#endif
#ifndef N_OUTER
#define N_OUTER 50
#endif

extern "C" __global__ __launch_bounds__(64, 8)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    int tid = threadIdx.x;
    int warp = tid / 32, lane = tid & 31;
    int bid = blockIdx.x;
    int* workspace = (int*)A;
    unsigned int* sm_pair = (unsigned int*)(A + (256 * 1024 * 1024 - 4096));
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

    int role = -1;
    if (warp == 0 && lane == 0) {
        for (int s = 0; s < N_LOADERS; s++) {
            unsigned int prev = atomicCAS(&sm_pair[smid * 16 + s], 0u, (unsigned)bid + 1);
            if (prev == 0u) { role = 0; break; }
        }
        if (role == -1) {
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
            unsigned int base = ((unsigned)(v + it + smid * 1337 + bid * 7) * 0x9E3779B1u) & 0x0FFFFF00u;
            #pragma unroll
            for (int k = 0; k < N_OPS; k++) {
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
    }
    if (role == 1 && warp == 0 && lane == 0) {
        ((unsigned long long*)C)[1024 + smid * 2] = (unsigned long long)total_dt;
    }
}
