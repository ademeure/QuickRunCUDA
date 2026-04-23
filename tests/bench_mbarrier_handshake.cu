// Producer-consumer handshake comparison: mbarrier vs bar.sync vs fence-poll.
// 2 warps in 1 CTA. Warp 0 = producer, warp 1 = consumer.
// Producer writes a value to smem and signals; consumer waits and reads.
// Measure consumer's "signal-to-data-available" latency.

#ifndef MODE
// 0 = mbarrier (init+arrive on producer, try_wait on consumer)
// 1 = bar.sync 0 (full CTA-wide sync on both warps)
// 2 = bar.arrive 1,64 + bar.sync 1,64 (split barrier)
// 3 = fence.release.cta + flag-poll + fence.acquire.cta (manual)
// 4 = mbarrier with cross-warp handshake (producer-side init, consumer-side wait)
#define MODE 0
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif

extern "C" __global__ __launch_bounds__(64, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (blockIdx.x != 0) return;
    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid & 31;

    __shared__ __align__(8) unsigned long long mb;
    __shared__ volatile int data;
    __shared__ volatile int flag;

    unsigned int mb_addr = __cvta_generic_to_shared(&mb);

    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mb_addr));
        flag = 0;
        data = 0;
    }
    __syncthreads();

    unsigned long long t0 = 0, t1 = 0;
    long long total_dt = 0;
    unsigned long long state = 0;
    int read_val = 0;

    for (int it = 0; it < N_OUTER; it++) {
        if (warp == 0 && lane == 0) {
            // Producer
#if MODE == 0 || MODE == 4
            // Reset mbarrier
            asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mb_addr));
            __threadfence_block();
            data = it + 1;
            asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(mb_addr));
#elif MODE == 1
            data = it + 1;
            asm volatile("bar.sync 0;" ::: "memory");
#elif MODE == 2
            data = it + 1;
            asm volatile("bar.arrive 1, 64;");
            asm volatile("bar.sync 1, 64;");
#elif MODE == 3
            data = it + 1;
            asm volatile("fence.release.cta;" ::: "memory");
            flag = it + 1;
#endif
        } else if (warp == 1 && lane == 0) {
            // Consumer
#if MODE == 0
            // Get state (producer initialized to expect 1 arrival per iter, but that's racy without init coordination — handled by producer)
            // For simplicity, just wait on the barrier with state=0 (should match parity)
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            int done = 0;
            while (done == 0) {
                int pred;
                asm volatile("{ .reg .pred %%p;\n"
                             "  mbarrier.try_wait.shared.b64 %%p, [%1], %2;\n"
                             "  selp.b32 %0, 1, 0, %%p; }"
                             : "=r"(pred) : "r"(mb_addr), "l"((unsigned long long)0));
                done = pred;
            }
            read_val ^= data;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
#elif MODE == 1
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            asm volatile("bar.sync 0;" ::: "memory");
            read_val ^= data;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
#elif MODE == 2
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            asm volatile("bar.arrive 1, 64;");
            asm volatile("bar.sync 1, 64;");
            read_val ^= data;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
#elif MODE == 3
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            // Spin on flag
            int target = it + 1;
            while (flag != target) {}
            asm volatile("fence.acquire.cta;" ::: "memory");
            read_val ^= data;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
#elif MODE == 4
            // Same as MODE 0 but with explicit producer-style init for next iter
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            int done = 0;
            while (done == 0) {
                int pred;
                asm volatile("{ .reg .pred %%p;\n"
                             "  mbarrier.try_wait.shared.b64 %%p, [%1], %2;\n"
                             "  selp.b32 %0, 1, 0, %%p; }"
                             : "=r"(pred) : "r"(mb_addr), "l"((unsigned long long)0));
                done = pred;
            }
            read_val ^= data;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
#endif
            total_dt += (long long)(t1 - t0);
        }
    }

    if (warp == 1 && lane == 0) {
        ((unsigned long long*)C)[1024] = (unsigned long long)total_dt;
        if (read_val == 0xDEADBEEF) C[0] = (float)read_val;
    }
}
