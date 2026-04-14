// All MUFU variants — latency + throughput.
// MODE=0: latency (1 thread, clock64, chain).
// MODE=1: throughput (grid, 8 chains).

#ifndef N_OPS
#define N_OPS 128
#endif
#ifndef ITERS_OUTER
#define ITERS_OUTER 256
#endif
#ifndef OP
#define OP 0
#endif
#ifndef MODE
#define MODE 0
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef CHAINS
#define CHAINS 8
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

#define DO_OP(F) \
  do { \
    if (OP == 0) asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(F)); \
    else if (OP == 1) asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(F)); \
    else if (OP == 2) asm volatile("sqrt.approx.f32 %0, %0;" : "+f"(F)); \
    else if (OP == 3) asm volatile("sqrt.approx.ftz.f32 %0, %0;" : "+f"(F)); \
    else if (OP == 4) asm volatile("lg2.approx.f32 %0, %0;" : "+f"(F)); \
    else if (OP == 5) asm volatile("lg2.approx.ftz.f32 %0, %0;" : "+f"(F)); \
    else if (OP == 6) asm volatile("sin.approx.f32 %0, %0;" : "+f"(F)); \
    else if (OP == 7) asm volatile("sin.approx.ftz.f32 %0, %0;" : "+f"(F)); \
    else if (OP == 8) asm volatile("cos.approx.f32 %0, %0;" : "+f"(F)); \
    else if (OP == 9) asm volatile("cos.approx.ftz.f32 %0, %0;" : "+f"(F)); \
    else if (OP == 10) asm volatile("rcp.approx.f32 %0, %0;" : "+f"(F)); \
    else if (OP == 11) asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(F)); \
  } while (0)

#if MODE == 0
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float f = 1.2345f;
    unsigned long long total_dt = 0;
    #pragma unroll 1
    for (int outer = 0; outer < ITERS_OUTER; outer++) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll
        for (int j = 0; j < N_OPS; j++) {
            DO_OP(f);
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total_dt += (t1 - t0);
    }
    if (__float_as_int(f) == seed) ((float*)C)[0] = f;
    ((unsigned long long*)C)[1] = total_dt;
}
#else
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float f[CHAINS];
    #pragma unroll
    for (int k=0;k<CHAINS;k++) f[k] = 1.2345f + 0.01f * k;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k=0;k<CHAINS;k++) DO_OP(f[k]);
        }
    }
    float acc = 0;
    #pragma unroll
    for (int k=0;k<CHAINS;k++) acc += f[k];
    if (__float_as_int(acc) == seed) ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
#endif
