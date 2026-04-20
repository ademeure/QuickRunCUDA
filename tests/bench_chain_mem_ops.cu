// Chain latency for memory + tensor ops vs FFMA.
// Single warp, 1000-deep chain through one register, anti-DCE.
//
// MODE codes:
//   0 = FFMA chain (baseline)
//   1 = LDG.E.STRONG.SM chain (each LDG addr depends on prev result)
//   2 = LDS chain (SHMEM read, addr depends on prev)
//   3 = LDG -> FFMA -> LDG -> FFMA pattern
//   4 = LDS -> FFMA -> LDS -> FFMA pattern
//   5 = mma.sync m16n8k16 BF16 chain (chain via accumulator)
//   6 = mma.sync -> FFMA (read scalar from acc) -> mma.sync

#include <cuda_bf16.h>

#ifndef N_INNER
#define N_INNER 200
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    // Initialize SHMEM region for LDS test
    __shared__ unsigned int smem[1024];
    if (threadIdx.x < 32) {
        for (int i = threadIdx.x; i < 1024; i += 32) smem[i] = i ^ 0xCAFEBABEu;
    }
    __syncwarp();

    // Initialize global memory hot region for LDG test
    unsigned int* Au = (unsigned int*)A;
    if (threadIdx.x == 0) {
        for (int i = 0; i < 1024; i++) Au[i] = i ^ 0xDEADBEEFu;
    }
    __syncwarp();

    float fv = (float)threadIdx.x + 1.5f;
    float fb = 1.0000001f + (float)u2 * 1e-9f;
    float fc = 0.0000001f + (float)u2 * 1e-9f;
    unsigned int v = (unsigned)threadIdx.x + 1u;
    unsigned int u2_pert = (unsigned)u2;  // for chain perturbation

    // mma.sync state
    unsigned int a0 = 0x3F803F80u, a1 = 0x3F803F80u, a2 = 0x3F803F80u, a3 = 0x3F803F80u;
    unsigned int b0 = 0x3F803F80u, b1 = 0x3F803F80u;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
#if MODE == 0
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
#elif MODE == 1
            // LDG chain: idx depends on prev v -> serial dep
            unsigned int idx = (v & 0x3FF) ^ u2_pert;
            unsigned int x;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(Au + idx));
            v = x;
#elif MODE == 2
            // LDS chain
            unsigned int idx = (v & 0x3FF) ^ u2_pert;
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];"
                         : "=r"(x) : "r"(__cvta_generic_to_shared(smem + idx)));
            v = x;
#elif MODE == 3
            // LDG -> FFMA -> LDG -> FFMA pattern (4 inst per iter)
            unsigned int idx = (v & 0x3FF) ^ u2_pert;
            unsigned int x;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(Au + idx));
            v = x;
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(__uint_as_float(v)), "f"(fc));
            v ^= __float_as_uint(fv);
            idx = (v & 0x3FF) ^ u2_pert;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(Au + idx));
            v = x;
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(__uint_as_float(v)), "f"(fc));
            v ^= __float_as_uint(fv);
#elif MODE == 4
            // LDS -> FFMA pattern
            unsigned int idx = (v & 0x3FF) ^ u2_pert;
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];"
                         : "=r"(x) : "r"(__cvta_generic_to_shared(smem + idx)));
            v = x;
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(__uint_as_float(v)), "f"(fc));
            v ^= __float_as_uint(fv);
            idx = (v & 0x3FF) ^ u2_pert;
            asm volatile("ld.shared.u32 %0, [%1];"
                         : "=r"(x) : "r"(__cvta_generic_to_shared(smem + idx)));
            v = x;
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(__uint_as_float(v)), "f"(fc));
            v ^= __float_as_uint(fv);
#elif MODE == 5
            // mma.sync chain via accumulator
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif MODE == 6
            // mma.sync -> FFMA -> mma.sync (FFMA reads c0)
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(c0) : "f"(fb), "f"(fc));
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed && (int)fv == seed && (int)c0 == seed)
        ((unsigned*)C)[blockIdx.x] = v + (unsigned)c0;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long pairs = (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
#if MODE == 3 || MODE == 4
        unsigned long long inst = pairs * 4;
#elif MODE == 6
        unsigned long long inst = pairs * 2;
#else
        unsigned long long inst = pairs;
#endif
        printf("MODE=%d inst=%llu clk=%llu cy/inst=%.3f cy/iter=%.3f\n",
               MODE, inst, t1 - t0,
               (double)(t1-t0)/(double)inst, (double)(t1-t0)/(double)pairs);
    }
}
