// Does TMA in flight compete with FFMA throughput?
// OP 0 = FFMA-only baseline
// OP 1 = FFMA + continuous TMA background (issue + wait loop in lane 0 only)
// OP 2 = LDG-streaming + FFMA baseline
// OP 3 = TMA pure (for reference)

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef OP
#define OP 0
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 65536
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long mbar;
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);
    unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);

    if (threadIdx.x == 0)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_addr));
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    float f = (float)(threadIdx.x + 1) * 0.0001f;
    unsigned long long t0 = 0, t1 = 0;
    unsigned phase = 0;

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if OP == 0 || OP == 2  // FFMA chain in all threads
        #pragma unroll 16
        for (int k = 0; k < 16; k++) f = f * 1.0000001f + 0.0000001f;
#endif

#if OP == 1 || OP == 3  // TMA (lane 0 only)
        if (threadIdx.x == 0) {
            unsigned long long src = (unsigned long long)A + ((i & 0xFF) << 16);
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbar_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1], %3, [%2];"
                :: "r"(smem_addr), "l"(src), "r"(mbar_addr),
                   "n"((unsigned)TMA_BYTES) : "memory");
            unsigned p = 0;
            for (int t = 0; t < 100000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(mbar_addr), "r"(phase));
            }
            phase ^= 1;
        }
#endif

#if OP == 1  // FFMA in parallel with TMA (all lanes)
        #pragma unroll 16
        for (int k = 0; k < 16; k++) f = f * 1.0000001f + 0.0000001f;
#endif

#if OP == 2  // LDG stream baseline (lane 0 only, to compare with TMA load)
        if (threadIdx.x == 0) {
            unsigned int v[8];
            unsigned long long src = (unsigned long long)A + ((i & 0xFF) << 16);
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(v[0]),"=r"(v[1]),"=r"(v[2]),"=r"(v[3])
                    : "l"(src + k*16));
                f += __int_as_float((int)(v[0]^v[1]^v[2]^v[3])) * 1e-30f;
            }
        }
#endif
    }

    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[blockIdx.x] = t1 - t0;
    }
    if ((int)__float_as_int(f) == seed) C[blockIdx.x * BLOCK_SIZE + threadIdx.x + 1024] = f;
}
