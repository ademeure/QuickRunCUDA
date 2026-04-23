// Test: does cp.async bypass fence drain?

#ifndef SLEEP_NS
#define SLEEP_NS 0
#endif
#ifndef N_LOADS
#define N_LOADS 16
#endif
#ifndef N_OUTER
#define N_OUTER 50
#endif
#ifndef MODE
// 0 = cp.async + acquire.gpu, no commit/wait
// 1 = cp.async + commit_group + acquire.gpu (no wait)
// 2 = cp.async + commit_group + wait_group + acquire.gpu
// 3 = cp.async BIG WS DRAM-cold + acquire.gpu
// 4 = cp.async + release.gpu
// 5 = cp.async + acquire.cta
// 6 = baseline: ld.volatile + acquire.gpu
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int* workspace = (int*)A;

    __shared__ __align__(16) int smem[1024];

    // Get shared memory address as PTX uint32
    unsigned int smem_addr = __cvta_generic_to_shared(smem);

    unsigned long long t0 = 0, t1 = 0;
    long long total_dt = 0;

    for (int it = 0; it < N_OUTER; it++) {
        // Address pattern: 16-byte aligned, varies per iter
        unsigned int base_off;

#if MODE == 6
        // Baseline: regular ld.volatile + acquire.gpu (DRAM-cold)
        unsigned int big_base = ((unsigned)(threadIdx.x + 1 + it) * 0x9E3779B1u) & 0x0FFFFFFCu;
        int x;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(x) : "l"(workspace + (big_base >> 2)));
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        if (x == 0xDEADBEEF) C[0] = (float)x;
        total_dt += (long long)(t1 - t0);

#elif MODE == 3
        // cp.async LARGE WS (DRAM-cold)
        base_off = ((unsigned)(threadIdx.x + 1 + it) * 0x9E3779B1u) & 0x0FFFFFF0u;
        #pragma unroll
        for (int k = 0; k < N_LOADS; k++) {
            unsigned int g_addr_b = (base_off + k * 4096) & 0x0FFFFFF0u;  // 16B aligned
            unsigned int s_addr = (smem_addr + ((k * 16) & 0xFF0)) & 0xFFFFFF0u;
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(s_addr), "l"((unsigned long long)workspace + g_addr_b)
            );
        }
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        asm volatile("fence.acquire.gpu;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncwarp();

#else
        // MODES 0, 1, 2, 4, 5: small WS, vary fence type and commit/wait
        base_off = ((unsigned)(threadIdx.x + 1 + it) * 0x9E3779B1u) & 0x00FFFFF0u;  // 16-byte aligned
        #pragma unroll
        for (int k = 0; k < N_LOADS; k++) {
            unsigned int g_addr_b = (base_off + k * 256) & 0x00FFFFF0u;
            unsigned int s_addr = (smem_addr + ((k * 16) & 0xFF0)) & 0xFFFFFF0u;
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(s_addr), "l"((unsigned long long)workspace + g_addr_b)
            );
        }

    #if MODE == 1 || MODE == 2
        asm volatile("cp.async.commit_group;" ::: "memory");
    #endif
    #if MODE == 2
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncwarp();
    #endif

        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)SLEEP_NS));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #if MODE == 0 || MODE == 1 || MODE == 2
        asm volatile("fence.acquire.gpu;" ::: "memory");
    #elif MODE == 4
        asm volatile("fence.release.gpu;" ::: "memory");
    #elif MODE == 5
        asm volatile("fence.acquire.cta;" ::: "memory");
    #endif

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        total_dt += (long long)(t1 - t0);
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncwarp();
#endif
    }

    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[1024] = (unsigned long long)total_dt;
    }
    if (threadIdx.x == 0 && smem[0] == 0xDEADBEEF) C[1] = (float)smem[0];
}
