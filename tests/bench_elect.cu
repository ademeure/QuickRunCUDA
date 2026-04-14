// Warp-specialization: elect.sync + work-on-leader vs no-elect.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = threadIdx.x;
    unsigned int sbase = (unsigned)__cvta_generic_to_shared(&smem[threadIdx.x]);
    if (threadIdx.x < 512) smem[threadIdx.x] = threadIdx.x;
    __syncthreads();

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // All lanes do work (baseline FFMA)
            float f = __int_as_float(v);
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f) : "f"(1.0001f), "f"(0.0001f));
            v = __float_as_int(f);
#elif OP == 1  // Predicated: only lane 0 does work
            if (threadIdx.x == 0) {
                float f = __int_as_float(v);
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f) : "f"(1.0001f), "f"(0.0001f));
                v = __float_as_int(f);
            }
#elif OP == 2  // elect.sync: only elected lane does work
            unsigned int leader;
            asm volatile("{.reg .pred p; elect.sync %0|p, 0xFFFFFFFF; @p mov.u32 %0, 1; @!p mov.u32 %0, 0;}" : "=r"(leader));
            if (leader) {
                float f = __int_as_float(v);
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f) : "f"(1.0001f), "f"(0.0001f));
                v = __float_as_int(f);
            }
#elif OP == 3  // All lanes do different work via shuffle-broadcast + process
            float f = __int_as_float(v);
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f) : "f"(1.0001f), "f"(0.0001f));
            float fb = __shfl_sync(0xFFFFFFFF, f, 0);
            v = __float_as_int(fb);
#elif OP == 4  // Warp specialization: lane 0 does load, others compute
            if (threadIdx.x == 0) {
                unsigned int r;
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(sbase));
                v = r;
            } else {
                float f = __int_as_float(v);
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f) : "f"(1.0001f), "f"(0.0001f));
                v = __float_as_int(f);
            }
#endif
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
