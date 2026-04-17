// Single-SM TMA efficiency sweep. Goal: find (size, parallel TMAs, pipeline depth)
// that saturate the ~196 GB/s per-SM TMA engine. All TMAs stomp same smem (SMEM_STRIDE=0)
// to avoid smem-cap issues. Forced ld.shared data dependency after each wait.
//
// Config:
//   TMA_BYTES  — size per TMA (128 B … 128 KB, 16-byte multiples)
//   NTMAS      — TMAs per mbarrier burst (1..16)
//   BARRIERS   — in-flight mbarriers round-robin (1..16)
// Each iter issues BARRIERS * NTMAS TMAs total; iter count = ITERS.
// Working set = ITERS * BARRIERS * NTMAS * TMA_BYTES bytes of A touched.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef NTMAS
#define NTMAS 4
#endif
#ifndef BARRIERS
#define BARRIERS 1
#endif
#ifndef STRIDE
#define STRIDE 524288
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bars[16];
    unsigned bar_base = (unsigned)__cvta_generic_to_shared(&bars[0]);
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int b = 0; b < BARRIERS; b++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_base + b*8));
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned long long t0 = 0, t1 = 0;
    unsigned int data_xor = 0;

    if (threadIdx.x == 0) {
        unsigned phase[16] = {0};
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

        #pragma unroll 1
        for (int i = 0; i < ITERS; i++) {
            int slot = i & (BARRIERS - 1);
            unsigned long long off = ((unsigned long long)i * STRIDE) & 0x3FFFFFFFull;
            unsigned long long src_base = (unsigned long long)A + off;

            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(bar_base + slot*8),
                   "n"((unsigned)(NTMAS * TMA_BYTES)) : "memory");
            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %3, [%2];"
                    :: "r"(smem_addr),  // all TMAs share smem region (SMEM_STRIDE=0)
                       "l"(src_base + k * TMA_BYTES),
                       "r"(bar_base + slot*8),
                       "n"((unsigned)TMA_BYTES) : "memory");
            }
            // If pipeline >= BARRIERS iters deep, wait on the one from BARRIERS iters back
            if (i >= BARRIERS - 1) {
                int old = (i + 1) & (BARRIERS - 1);  // oldest in-flight
                if (BARRIERS == 1) old = 0;
                unsigned p = 0;
                for (int t = 0; t < 10000 && !p; t++) {
                    asm volatile(
                        "{ .reg .pred P; "
                        "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                        "selp.b32 %0, 1, 0, P; }"
                        : "=r"(p) : "r"(bar_base + old*8), "r"(phase[old]));
                }
                phase[old] ^= 1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                unsigned int x;
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem_addr));
                data_xor ^= x;
            }
        }
        // drain
        for (int s = 0; s < BARRIERS; s++) {
            unsigned p = 0;
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar_base + s*8), "r"(phase[s]));
            }
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned int*)C)[4] = data_xor;
    }
}
