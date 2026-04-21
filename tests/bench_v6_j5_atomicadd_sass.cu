// V6 J5: Compare C++ atomicAdd vs raw PTX atom.shared.add SASS
// MODE 0: C++ atomicAdd on __shared__ uint
// MODE 1: Raw PTX atom.shared.add.u32
// MODE 2: C++ atomicAdd on global uint
// MODE 3: Raw PTX atom.global.add.u32
// Measure cy/iter for each, verify SASS opcode identity
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ unsigned int counter;
    if (threadIdx.x == 0) counter = 0;
    __syncwarp();

    unsigned int my_val = threadIdx.x + 1;
    unsigned int* g_counter = (unsigned int*)A;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        atomicAdd(&counter, my_val);
#elif MODE == 1
        unsigned int old;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;"
                     : "=r"(old) : "r"(__cvta_generic_to_shared(&counter)), "r"(my_val));
#elif MODE == 2
        atomicAdd(g_counter, my_val);
#elif MODE == 3
        unsigned int old;
        asm volatile("atom.global.add.u32 %0, [%1], %2;"
                     : "=r"(old) : "l"(g_counter), "r"(my_val));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (counter == 0xCAFEBABE) C[blockIdx.x] = 1.0f;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d cy/iter=%.3f\n", MODE, (double)(t1-t0)/(double)ITERS);
    }
}
