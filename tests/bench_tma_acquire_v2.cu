// Acquire pattern with tunable SRC_STRIDE so we can test L2-hot vs DRAM-miss.
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 65536
#endif
#ifndef DEPTH
#define DEPTH 3
#endif
#ifndef SRC_STRIDE
#define SRC_STRIDE TMA_BYTES
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long full[DEPTH];
    __shared__ __align__(8) unsigned long long empty[DEPTH];
    unsigned full_base  = (unsigned)__cvta_generic_to_shared(&full[0]);
    unsigned empty_base = (unsigned)__cvta_generic_to_shared(&empty[0]);
    unsigned smem_addr  = (unsigned)__cvta_generic_to_shared(smem_buf);

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int b = 0; b < DEPTH; b++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(full_base  + b*8));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(empty_base + b*8));
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(empty_base + b*8));
        }
    }
    __syncthreads();

    unsigned long long t0 = 0, t1 = 0;
    unsigned int data_xor = 0;
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (threadIdx.x == 0) {
        unsigned p_ph = 0;
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            unsigned target = (p_ph >> slot) & 1;
            unsigned p = 0;
            while (!p) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2, 10000; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(empty_base + slot*8), "r"(target));
            }
            p_ph ^= (1u << slot);
            unsigned long long off = ((unsigned long long)blockIdx.x * DEPTH * TMA_BYTES
                                    + (unsigned long long)i * SRC_STRIDE) & 0x3FFFFFFFull;
            unsigned long long src = (unsigned long long)A + off;
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(full_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %3, [%2];"
                :: "r"(smem_addr + slot * TMA_BYTES), "l"(src),
                   "r"(full_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
        }
    } else if (wid == 1 && lid == 0) {
        unsigned c_ph = 0;
        unsigned int local_xor = 0;
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            unsigned target = (c_ph >> slot) & 1;
            unsigned p = 0;
            while (!p) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2, 10000; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(full_base + slot*8), "r"(target));
            }
            c_ph ^= (1u << slot);
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem_addr + slot * TMA_BYTES));
            local_xor ^= x;
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(empty_base + slot*8));
        }
        data_xor = local_xor;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[blockIdx.x * 3 + 0] = t1 - t0;
    }
    if ((int)data_xor == seed)
        ((unsigned int*)C)[blockIdx.x * 1024 + threadIdx.x] = data_xor;
}
