// Does per-thread predication affect pipe throughput?

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
#define OP 0  // 0=FFMA unpred, 1=FFMA half predicated, 2=FFMA 1-lane predicated, 3=CREDUX+FMNMX mixed
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float f0=1.0001f+0.0001f*(threadIdx.x+0*23),f1=1.0001f+0.0001f*(threadIdx.x+1*23);
    float f2=1.0001f+0.0001f*(threadIdx.x+2*23),f3=1.0001f+0.0001f*(threadIdx.x+3*23);
    float f4=1.0001f+0.0001f*(threadIdx.x+4*23),f5=1.0001f+0.0001f*(threadIdx.x+5*23);
    float f6=1.0001f+0.0001f*(threadIdx.x+6*23),f7=1.0001f+0.0001f*(threadIdx.x+7*23);
    unsigned int u0=threadIdx.x, u1=threadIdx.x+1;
    (void)u0; (void)u1;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f0) : "f"(1.000001f), "f"(0.9999f));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f1) : "f"(1.000001f), "f"(0.9999f));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f2) : "f"(1.000001f), "f"(0.9999f));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f3) : "f"(1.000001f), "f"(0.9999f));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f4) : "f"(1.000001f), "f"(0.9999f));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f5) : "f"(1.000001f), "f"(0.9999f));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f6) : "f"(1.000001f), "f"(0.9999f));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f7) : "f"(1.000001f), "f"(0.9999f));
#elif OP == 1
            // Half the warp predicated off (lanes 16..31)
            asm volatile("{ .reg .pred p; setp.lt.u32 p, %3, 16; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f0) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
            asm volatile("{ .reg .pred p; setp.lt.u32 p, %3, 16; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f1) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
            asm volatile("{ .reg .pred p; setp.lt.u32 p, %3, 16; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f2) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
            asm volatile("{ .reg .pred p; setp.lt.u32 p, %3, 16; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f3) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
            asm volatile("{ .reg .pred p; setp.lt.u32 p, %3, 16; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f4) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
            asm volatile("{ .reg .pred p; setp.lt.u32 p, %3, 16; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f5) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
            asm volatile("{ .reg .pred p; setp.lt.u32 p, %3, 16; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f6) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
            asm volatile("{ .reg .pred p; setp.lt.u32 p, %3, 16; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f7) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
#elif OP == 2
            // Only lane 0 active
            asm volatile("{ .reg .pred p; setp.eq.u32 p, %3, 0; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f0) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
            asm volatile("{ .reg .pred p; setp.eq.u32 p, %3, 0; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f1) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
            asm volatile("{ .reg .pred p; setp.eq.u32 p, %3, 0; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f2) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
            asm volatile("{ .reg .pred p; setp.eq.u32 p, %3, 0; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f3) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
            asm volatile("{ .reg .pred p; setp.eq.u32 p, %3, 0; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f4) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
            asm volatile("{ .reg .pred p; setp.eq.u32 p, %3, 0; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f5) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
            asm volatile("{ .reg .pred p; setp.eq.u32 p, %3, 0; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f6) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
            asm volatile("{ .reg .pred p; setp.eq.u32 p, %3, 0; @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(f7) : "f"(1.000001f), "f"(0.9999f), "r"((unsigned)threadIdx.x & 31u));
#elif OP == 3
            // Mix CREDUX.MIN + FMNMX — both on pipe_alu; test contention
            asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(u0));
            asm volatile("min.f32 %0, %0, %1;" : "+f"(f0) : "f"(f1));
            asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(u0));
            asm volatile("min.f32 %0, %0, %1;" : "+f"(f2) : "f"(f3));
            asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(u0));
            asm volatile("min.f32 %0, %0, %1;" : "+f"(f4) : "f"(f5));
            asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(u0));
            asm volatile("min.f32 %0, %0, %1;" : "+f"(f6) : "f"(f7));
#endif
        }
    }
    float acc = f0+f1+f2+f3+f4+f5+f6+f7;
    if (__float_as_int(acc) == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
