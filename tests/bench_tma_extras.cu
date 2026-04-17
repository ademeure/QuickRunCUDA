// Final battery: stalled polling, deep pipeline, reduce ops, aligned/unaligned, smem→smem.
//
// OP 0  = stalled poll rate — time try_wait.parity on NEVER-completing barrier (must be bounded)
// OP 1  = stalled poll rate — test_wait (non-blocking false)
// OP 10-16 = cp.reduce bulk with each op/dtype combo
//   10 = add.u32, 11 = add.s32, 12 = min.u32, 13 = max.u32
//   14 = min.s32, 15 = max.s32, 16 = and.b32
// OP 20 = cp.reduce add.f32
// OP 30 = pipeline depth 4 barriers  (same NTMAS/sz as OP 31/32)
// OP 31 = pipeline depth 8
// OP 32 = pipeline depth 32
// OP 40 = mbarrier chain: arrive→wait→arrive→wait (serial dependence chain)
// OP 50 = aligned(+0) vs 16B-offset src

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif
#ifndef OP
#define OP 0
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef PIPE_DEPTH
#define PIPE_DEPTH 16
#endif
#ifndef OFFSET
#define OFFSET 0
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bars[64];
    __shared__ __align__(128) unsigned int smem_redsrc[1024];  // for reduce test
    unsigned bar_base = (unsigned)__cvta_generic_to_shared(&bars[0]);
    unsigned redsrc_addr = (unsigned)__cvta_generic_to_shared(smem_redsrc);
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int b = 0; b < 64; b++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_base + b*8));
        for (int i = 0; i < 1024; i++) smem_redsrc[i] = i + 1;
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned long long t0 = 0, t1 = 0;

    if (threadIdx.x == 0) {
        unsigned phases[64] = {0};
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

        #pragma unroll 1
        for (int i = 0; i < ITERS; i++) {
            unsigned long long off = ((unsigned long long)i * 128 + OFFSET) & 0x3FFFFFFFull;
            unsigned long long gaddr = (unsigned long long)A + off;
            unsigned long long baddr = (unsigned long long)B + off;

#if OP == 0  // try_wait.parity on STALLED barrier, poll-until-false-returns
            unsigned p;
            asm volatile(
                "{ .reg .pred P; "
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], 1; "  // phase=1, barrier still at 0
                "selp.b32 %0, 1, 0, P; }"
                : "=r"(p) : "r"(bar_base + 8));
            if (p) break;  // won't happen (stalled) but shows DCE doesn't skip

#elif OP == 1  // test_wait on STALLED
            unsigned p;
            asm volatile(
                "{ .reg .pred P; "
                "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], 1; "
                "selp.b32 %0, 1, 0, P; }"
                : "=r"(p) : "r"(bar_base + 8));
            if (p) break;

#elif OP == 10  // cp.reduce add.u32
            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], %2;"
                :: "l"(baddr), "r"(redsrc_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group 0;");
#elif OP == 11
            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.s32 [%0], [%1], %2;"
                :: "l"(baddr), "r"(redsrc_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;"); asm volatile("cp.async.bulk.wait_group 0;");
#elif OP == 12
            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.u32 [%0], [%1], %2;"
                :: "l"(baddr), "r"(redsrc_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;"); asm volatile("cp.async.bulk.wait_group 0;");
#elif OP == 13
            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.u32 [%0], [%1], %2;"
                :: "l"(baddr), "r"(redsrc_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;"); asm volatile("cp.async.bulk.wait_group 0;");
#elif OP == 14
            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s32 [%0], [%1], %2;"
                :: "l"(baddr), "r"(redsrc_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;"); asm volatile("cp.async.bulk.wait_group 0;");
#elif OP == 15
            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s32 [%0], [%1], %2;"
                :: "l"(baddr), "r"(redsrc_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;"); asm volatile("cp.async.bulk.wait_group 0;");
#elif OP == 16
            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.and.b32 [%0], [%1], %2;"
                :: "l"(baddr), "r"(redsrc_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;"); asm volatile("cp.async.bulk.wait_group 0;");
#elif OP == 17
            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.or.b32 [%0], [%1], %2;"
                :: "l"(baddr), "r"(redsrc_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;"); asm volatile("cp.async.bulk.wait_group 0;");
#elif OP == 18
            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b32 [%0], [%1], %2;"
                :: "l"(baddr), "r"(redsrc_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;"); asm volatile("cp.async.bulk.wait_group 0;");
#elif OP == 20
            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
                :: "l"(baddr), "r"(redsrc_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;"); asm volatile("cp.async.bulk.wait_group 0;");
#elif OP == 21
            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f16 [%0], [%1], %2;"
                :: "l"(baddr), "r"(redsrc_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;"); asm volatile("cp.async.bulk.wait_group 0;");
#elif OP == 22
            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.bf16 [%0], [%1], %2;"
                :: "l"(baddr), "r"(redsrc_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;"); asm volatile("cp.async.bulk.wait_group 0;");

#elif OP == 30  // PIPELINE_DEPTH in flight with per-TMA barrier
            int slot = i & (PIPE_DEPTH - 1);
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(bar_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1], %3, [%2];"
                :: "r"(smem_addr + slot * TMA_BYTES), "l"(gaddr),
                   "r"(bar_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
            if (i >= PIPE_DEPTH) {
                int old = (i - PIPE_DEPTH) & (PIPE_DEPTH - 1);
                unsigned p = 0;
                for (int t = 0; t < 1000 && !p; t++) {
                    asm volatile(
                        "{ .reg .pred P; "
                        "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                        "selp.b32 %0, 1, 0, P; }"
                        : "=r"(p) : "r"(bar_base + old*8), "r"(phases[old]));
                }
                phases[old] ^= 1;
            }

#elif OP == 40  // mbarrier chain — serial dep (arrive→wait→arrive→wait)
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar_base));
            unsigned p = 0;
            for (int t = 0; t < 100 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar_base), "r"(phases[0]));
            }
            phases[0] ^= 1;
#endif
        }

#if OP == 30
        for (int s = 0; s < PIPE_DEPTH; s++) {
            unsigned p = 0;
            for (int t = 0; t < 1000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar_base + s*8), "r"(phases[s]));
            }
        }
#endif

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();
    if (threadIdx.x == 0)
        ((unsigned long long*)C)[blockIdx.x] = t1 - t0;
}
