// mbarrier fan-in scaling: N threads arrive at one mbarrier (count=N).
// Measures barrier-RTT as function of arriving-thread count.
// Single CTA; sweep BLOCK_SIZE via -t.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bar;
    unsigned bar_addr = (unsigned)__cvta_generic_to_shared(&bar);

    if (threadIdx.x == 0)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "r"(bar_addr), "r"((unsigned)BLOCK_SIZE));
    __syncthreads();

    unsigned long long t0 = 0, t1 = 0;
    unsigned phase = 0;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if OP == 0  // all threads arrive, leader waits
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar_addr));
        if (threadIdx.x == 0) {
            unsigned p = 0;
            for (int t = 0; t < 1000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar_addr), "r"(phase));
            }
            phase ^= 1;
        }

#elif OP == 1  // all threads arrive + all threads wait
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar_addr));
        unsigned p = 0;
        for (int t = 0; t < 1000 && !p; t++) {
            asm volatile(
                "{ .reg .pred P; "
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                "selp.b32 %0, 1, 0, P; }"
                : "=r"(p) : "r"(bar_addr), "r"(phase));
        }
        phase ^= 1;

#elif OP == 2  // plain __syncthreads baseline
        __syncthreads();
#endif
    }

    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[blockIdx.x] = t1 - t0;
    }
}
