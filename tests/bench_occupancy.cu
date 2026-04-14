// Occupancy effects: does MIN_BLOCKS / BLOCK_SIZE affect per-warp FFMA throughput?

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef REG_PRESSURE
#define REG_PRESSURE 8     // number of independent FFMA chains
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float f[REG_PRESSURE];
    #pragma unroll
    for (int k=0;k<REG_PRESSURE;k++) f[k] = 1.0001f + 0.0001f*(threadIdx.x + k*23);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k=0;k<REG_PRESSURE;k++)
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[k]) : "f"(1.000001f), "f"(0.9999f));
        }
    }
    float acc = 0;
    #pragma unroll
    for (int k=0;k<REG_PRESSURE;k++) acc += f[k];
    if (__float_as_int(acc) == seed) ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
