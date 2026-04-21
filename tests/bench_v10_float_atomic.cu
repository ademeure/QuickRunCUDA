// V10: float vs int atomicAdd cost (single-thread chain)
#ifndef OP
#define OP 0  // 0=int_smem, 1=float_smem, 2=int_global, 3=float_global
#endif
#ifndef CHAIN_LEN
#define CHAIN_LEN 1000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;
    __shared__ unsigned int s_int[1];
    __shared__ float s_float[1];
    s_int[0] = 0;
    s_float[0] = 0.0f;

    unsigned int vi = 0;
    float vf = 0.0f;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
#if OP == 0
        vi = atomicAdd(s_int, vi + 1);
#elif OP == 1
        vf = atomicAdd(s_float, vf + 1.0f);
#elif OP == 2
        vi = atomicAdd((unsigned*)A, vi + 1);
#elif OP == 3
        vf = atomicAdd(A, vf + 1.0f);
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    ((unsigned long long*)C)[0] = t1 - t0;
    ((float*)C)[2] = vf + (float)vi;
}
