// Test if atomicAdd with VARIABLE value still gets POPC.INC optimization

#ifndef CONST_VAL
#define CONST_VAL 0  // 0 = use variable per-lane value, 1 = const 1
#endif
#ifndef CONTENTION
#define CONTENTION 0  // 0=uniq 1=broadcast
#endif
#ifndef ITERS
#define ITERS 1000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned int* A, float* B, float* C, int seed, int u1, int u2) {
    if (blockIdx.x != 0) return;
    int lane = threadIdx.x;

    __shared__ unsigned int smem[1024];
    if (lane == 0) for (int i = 0; i < 1024; i++) smem[i] = 0;
    __syncwarp();

    unsigned int* addr;
    #if CONTENTION == 0
        addr = &smem[lane * 32];
    #elif CONTENTION == 1
        addr = &smem[0];
    #endif

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if CONST_VAL == 1
        atomicAdd(addr, 1u);  // constant — POPC.INC eligible
#else
        atomicAdd(addr, (unsigned)(lane + i));  // variable — must use REDS.ADD
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (lane == 0) {
        ((unsigned long long*)C)[1024] = (unsigned long long)(t1 - t0);
    }
}
