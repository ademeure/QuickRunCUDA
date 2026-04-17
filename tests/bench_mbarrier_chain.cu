// mbarrier cost patterns — chained / fan-in / ping-pong / many barriers.
//
// OP 0 = serial arrive→try_wait on ONE mbarrier (count=1), phase flip each iter
// OP 1 = fan-in: all 32 lanes arrive once, try_wait.parity
// OP 2 = ping-pong between 2 mbarriers (arriveA→waitA→arriveB→waitB)
// OP 3 = 8-barrier round-robin
// OP 4 = chain N arrive.expect_tx → single wait at end (batched)
// OP 5 = single barrier, try_wait.parity on complete (no wait, fast path)
// OP 6 = test_wait on completed barrier (immediate)
// OP 7 = test_wait on incomplete barrier (immediate false)
// OP 8 = arrive + mbarrier.inval (teardown cost)
// OP 9 = try_wait.parity with suspendTimeHint=0 on completed

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bar[16];

    unsigned bar_base = (unsigned)__cvta_generic_to_shared(&bar[0]);

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int b = 0; b < 16; b++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                :: "r"(bar_base + b*8), "r"((unsigned)(OP == 1 ? BLOCK_SIZE : 1)));
        }
    }
    __syncthreads();

    unsigned long long t0 = 0, t1 = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    unsigned phase_a = 0, phase_b = 0;
    unsigned phase[8] = {0,0,0,0,0,0,0,0};

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
#if OP == 0 && BLOCK_SIZE == 1  // serial arrive+try_wait same barrier
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar_base));
            unsigned p = 0;
            for (int t = 0; t < 100 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar_base), "r"(phase_a));
            }
            phase_a ^= 1;

#elif OP == 1  // fan-in: full block arrives, leader waits
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar_base));
            if (threadIdx.x == 0) {
                unsigned p = 0;
                for (int t = 0; t < 100 && !p; t++) {
                    asm volatile(
                        "{ .reg .pred P; "
                        "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                        "selp.b32 %0, 1, 0, P; }"
                        : "=r"(p) : "r"(bar_base), "r"(phase_a));
                }
                phase_a ^= 1;
            }

#elif OP == 2 && BLOCK_SIZE == 1  // ping-pong A/B
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar_base));
            unsigned pa = 0;
            for (int t = 0; t < 100 && !pa; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(pa) : "r"(bar_base), "r"(phase_a));
            }
            phase_a ^= 1;
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar_base + 8));
            unsigned pb = 0;
            for (int t = 0; t < 100 && !pb; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(pb) : "r"(bar_base + 8), "r"(phase_b));
            }
            phase_b ^= 1;

#elif OP == 3 && BLOCK_SIZE == 1  // 8-barrier round-robin
            int slot = u & 7;
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar_base + slot*8));
            unsigned p = 0;
            for (int t = 0; t < 100 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar_base + slot*8), "r"(phase[slot]));
            }
            phase[slot] ^= 1;

#elif OP == 4 && BLOCK_SIZE == 1  // batch arrives, one wait (after UNROLL arrives)
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], 0;"
                :: "r"(bar_base));
            if (u == UNROLL - 1) {
                unsigned p = 0;
                for (int t = 0; t < 1000 && !p; t++) {
                    asm volatile(
                        "{ .reg .pred P; "
                        "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                        "selp.b32 %0, 1, 0, P; }"
                        : "=r"(p) : "r"(bar_base), "r"(phase_a));
                }
                phase_a ^= 1;
            }

#elif OP == 5  // try_wait.parity on completed (fast path, NO arrive)
            unsigned p;
            asm volatile(
                "{ .reg .pred P; "
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                "selp.b32 %0, 1, 0, P; }"
                : "=r"(p) : "r"(bar_base + 8), "r"(phase_b));

#elif OP == 6  // test_wait on completed
            unsigned p;
            asm volatile(
                "{ .reg .pred P; "
                "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                "selp.b32 %0, 1, 0, P; }"
                : "=r"(p) : "r"(bar_base + 8), "r"(phase_b));

#elif OP == 7  // test_wait on incomplete (returns false fast)
            unsigned p;
            asm volatile(
                "{ .reg .pred P; "
                "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                "selp.b32 %0, 1, 0, P; }"
                : "=r"(p) : "r"(bar_base + 16), "r"(phase_a));

#elif OP == 8 && BLOCK_SIZE == 1  // init + inval cost
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_base + 16));
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(bar_base + 16));

#elif OP == 9  // try_wait.parity with suspendTimeHint
            unsigned p;
            asm volatile(
                "{ .reg .pred P; "
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2, %3; "
                "selp.b32 %0, 1, 0, P; }"
                : "=r"(p) : "r"(bar_base + 8), "r"(phase_b), "r"(0u));
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    __syncthreads();
    if (threadIdx.x == 0)
        ((unsigned long long*)C)[blockIdx.x] = t1 - t0;
}
