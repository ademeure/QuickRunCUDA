// Deep atomics test — per-lane distinct addresses so rate measures pipe,
// not contention. Each thread has its own slot in shared memory.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v0 = threadIdx.x, v1 = threadIdx.x + 1, v2 = threadIdx.x + 2, v3 = threadIdx.x + 3;
    unsigned int v4 = threadIdx.x + 4, v5 = threadIdx.x + 5, v6 = threadIdx.x + 6, v7 = threadIdx.x + 7;
    // Each thread owns slot [threadIdx.x*8 + k] — no cross-lane conflict
    unsigned int base = (unsigned)__cvta_generic_to_shared(&smem[threadIdx.x * 8]);
    if (threadIdx.x < BLOCK_SIZE) {
        for (int k = 0; k < 8; k++) smem[threadIdx.x * 8 + k] = threadIdx.x;
    }
    __syncthreads();

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#define A8(OPASM) \
    asm volatile(OPASM : "+r"(v0) : "r"(base + 0*4), "r"(v0)); \
    asm volatile(OPASM : "+r"(v1) : "r"(base + 1*4), "r"(v1)); \
    asm volatile(OPASM : "+r"(v2) : "r"(base + 2*4), "r"(v2)); \
    asm volatile(OPASM : "+r"(v3) : "r"(base + 3*4), "r"(v3)); \
    asm volatile(OPASM : "+r"(v4) : "r"(base + 4*4), "r"(v4)); \
    asm volatile(OPASM : "+r"(v5) : "r"(base + 5*4), "r"(v5)); \
    asm volatile(OPASM : "+r"(v6) : "r"(base + 6*4), "r"(v6)); \
    asm volatile(OPASM : "+r"(v7) : "r"(base + 7*4), "r"(v7));

#if OP == 0   // atom.shared.min.u32
            A8("atom.shared.min.u32 %0, [%1], %2;")
#elif OP == 1 // atom.shared.max.u32
            A8("atom.shared.max.u32 %0, [%1], %2;")
#elif OP == 2 // atom.shared.add.u32
            A8("atom.shared.add.u32 %0, [%1], %2;")
#elif OP == 3 // atom.shared.and.b32
            A8("atom.shared.and.b32 %0, [%1], %2;")
#elif OP == 4 // atom.shared.or.b32
            A8("atom.shared.or.b32 %0, [%1], %2;")
#elif OP == 5 // atom.shared.xor.b32
            A8("atom.shared.xor.b32 %0, [%1], %2;")
#elif OP == 6 // atom.shared.exch.b32
            A8("atom.shared.exch.b32 %0, [%1], %2;")
#elif OP == 7 // atom.shared.inc.u32
            A8("atom.shared.inc.u32 %0, [%1], %2;")
#elif OP == 8 // atom.shared.dec.u32
            A8("atom.shared.dec.u32 %0, [%1], %2;")
#elif OP == 9 // atom.shared.cas.b32 — uses compare + swap
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "+r"(v0) : "r"(base + 0*4), "r"(v0), "r"(v1));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "+r"(v1) : "r"(base + 1*4), "r"(v1), "r"(v2));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "+r"(v2) : "r"(base + 2*4), "r"(v2), "r"(v3));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "+r"(v3) : "r"(base + 3*4), "r"(v3), "r"(v4));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "+r"(v4) : "r"(base + 4*4), "r"(v4), "r"(v5));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "+r"(v5) : "r"(base + 5*4), "r"(v5), "r"(v6));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "+r"(v6) : "r"(base + 6*4), "r"(v6), "r"(v7));
            asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "+r"(v7) : "r"(base + 7*4), "r"(v7), "r"(v0));
#elif OP == 10 // red.shared.add.u32 (no return — possibly faster)
            asm volatile("red.shared.add.u32 [%0], %1;" :: "r"(base + 0*4), "r"(v0));
            asm volatile("red.shared.add.u32 [%0], %1;" :: "r"(base + 1*4), "r"(v1));
            asm volatile("red.shared.add.u32 [%0], %1;" :: "r"(base + 2*4), "r"(v2));
            asm volatile("red.shared.add.u32 [%0], %1;" :: "r"(base + 3*4), "r"(v3));
            asm volatile("red.shared.add.u32 [%0], %1;" :: "r"(base + 4*4), "r"(v4));
            asm volatile("red.shared.add.u32 [%0], %1;" :: "r"(base + 5*4), "r"(v5));
            asm volatile("red.shared.add.u32 [%0], %1;" :: "r"(base + 6*4), "r"(v6));
            asm volatile("red.shared.add.u32 [%0], %1;" :: "r"(base + 7*4), "r"(v7));
#elif OP == 11 // atom.shared.add.u64
            { unsigned long long lv0=v0,lv1=v1,lv2=v2,lv3=v3,lv4=v4,lv5=v5,lv6=v6,lv7=v7;
              asm volatile("atom.shared.add.u64 %0, [%1], %2;" : "+l"(lv0) : "r"(base + 0*4), "l"(lv0));
              asm volatile("atom.shared.add.u64 %0, [%1], %2;" : "+l"(lv1) : "r"(base + 1*4), "l"(lv1));
              asm volatile("atom.shared.add.u64 %0, [%1], %2;" : "+l"(lv2) : "r"(base + 2*4), "l"(lv2));
              asm volatile("atom.shared.add.u64 %0, [%1], %2;" : "+l"(lv3) : "r"(base + 3*4), "l"(lv3));
              asm volatile("atom.shared.add.u64 %0, [%1], %2;" : "+l"(lv4) : "r"(base + 4*4), "l"(lv4));
              asm volatile("atom.shared.add.u64 %0, [%1], %2;" : "+l"(lv5) : "r"(base + 5*4), "l"(lv5));
              asm volatile("atom.shared.add.u64 %0, [%1], %2;" : "+l"(lv6) : "r"(base + 6*4), "l"(lv6));
              asm volatile("atom.shared.add.u64 %0, [%1], %2;" : "+l"(lv7) : "r"(base + 7*4), "l"(lv7));
              v0=(unsigned)lv0; v1=(unsigned)lv1; v2=(unsigned)lv2; v3=(unsigned)lv3;
              v4=(unsigned)lv4; v5=(unsigned)lv5; v6=(unsigned)lv6; v7=(unsigned)lv7; }
#elif OP == 12 // atom.shared.add.f32
            { float f0=__int_as_float(v0),f1=__int_as_float(v1),f2=__int_as_float(v2),f3=__int_as_float(v3);
              float f4=__int_as_float(v4),f5=__int_as_float(v5),f6=__int_as_float(v6),f7=__int_as_float(v7);
              asm volatile("atom.shared.add.f32 %0, [%1], %2;" : "+f"(f0) : "r"(base + 0*4), "f"(f0));
              asm volatile("atom.shared.add.f32 %0, [%1], %2;" : "+f"(f1) : "r"(base + 1*4), "f"(f1));
              asm volatile("atom.shared.add.f32 %0, [%1], %2;" : "+f"(f2) : "r"(base + 2*4), "f"(f2));
              asm volatile("atom.shared.add.f32 %0, [%1], %2;" : "+f"(f3) : "r"(base + 3*4), "f"(f3));
              asm volatile("atom.shared.add.f32 %0, [%1], %2;" : "+f"(f4) : "r"(base + 4*4), "f"(f4));
              asm volatile("atom.shared.add.f32 %0, [%1], %2;" : "+f"(f5) : "r"(base + 5*4), "f"(f5));
              asm volatile("atom.shared.add.f32 %0, [%1], %2;" : "+f"(f6) : "r"(base + 6*4), "f"(f6));
              asm volatile("atom.shared.add.f32 %0, [%1], %2;" : "+f"(f7) : "r"(base + 7*4), "f"(f7));
              v0=__float_as_int(f0);v1=__float_as_int(f1);v2=__float_as_int(f2);v3=__float_as_int(f3);
              v4=__float_as_int(f4);v5=__float_as_int(f5);v6=__float_as_int(f6);v7=__float_as_int(f7); }
#elif OP == 13 // atom.global.add.u32 — per-thread addresses
            A8("atom.global.add.u32 %0, [%1], %2;")
#endif
        }
    }
    unsigned int acc = v0^v1^v2^v3^v4^v5^v6^v7;
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
