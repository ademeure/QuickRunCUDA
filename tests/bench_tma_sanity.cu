// Sanity: TMA load, wait, then read smem[0..7] and write to C.
// If the wait is fake, we'll see garbage. If not, we see expected bytes.

#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long mbar;
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);
    unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);

    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_addr));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned phase = 0;
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "n"((unsigned)TMA_BYTES) : "memory");
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %3, [%2];"
        :: "r"(smem_addr), "l"((unsigned long long)A), "r"(mbar_addr),
           "n"((unsigned)TMA_BYTES) : "memory");

    unsigned p = 0;
    for (int t = 0; t < 100000 && !p; t++) {
        asm volatile(
            "{ .reg .pred P; "
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
            "selp.b32 %0, 1, 0, P; }"
            : "=r"(p) : "r"(mbar_addr), "r"(phase));
    }

    // Force data dependency
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    unsigned int v0, v1, v2, v3;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3) : "r"(smem_addr));

    C[0] = __int_as_float((int)v0);
    C[1] = __int_as_float((int)v1);
    C[2] = __int_as_float((int)v2);
    C[3] = __int_as_float((int)v3);
    C[4] = __int_as_float((int)p);  // did the wait succeed?
}
