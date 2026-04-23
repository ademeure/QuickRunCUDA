// Test cluster-scope fences (B300/Hopper+)
#ifndef MODE
// 0 = idle + fence.acquire.cluster
// 1 = idle + fence.release.cluster
// 2 = LD L2 + acquire.cluster
// 3 = LD L2 + release.cluster
// 4 = ST + release.cluster
#define MODE 0
#endif
#ifndef N_OUTER
#define N_OUTER 50
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int* workspace = (int*)A;
    unsigned long long t0 = 0, t1 = 0;
    long long total_dt = 0;
    int v = threadIdx.x + 1;

    for (int it = 0; it < N_OUTER; it++) {
#if MODE == 0
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)2000));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.cluster;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
#elif MODE == 1
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)2000));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.cluster;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
#elif MODE == 2
        int x;
        unsigned int addr = ((unsigned)(v + it) * 64u) & 0x000FFFFCu;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(x) : "l"(workspace + (addr >> 2)));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.cluster;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= x;
#elif MODE == 3
        int x;
        unsigned int addr = ((unsigned)(v + it) * 64u) & 0x000FFFFCu;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(x) : "l"(workspace + (addr >> 2)));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.cluster;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        v ^= x;
#elif MODE == 4
        unsigned int addr = ((unsigned)(v + it) * 64u) & 0x000FFFFCu;
        int sv = v + (int)it;
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(workspace + (addr >> 2)), "r"(sv) : "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.release.cluster;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
#endif
        total_dt += (long long)(t1 - t0);
    }
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[1024] = (unsigned long long)total_dt;
        C[0] = (float)v;  // anti-DCE
    }
}
