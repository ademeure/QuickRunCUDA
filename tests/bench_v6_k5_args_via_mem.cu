// V6 K5: Kernel arg passed via cmem (kernel param) vs via global memory pointer
// MODE 0: pass int directly (in cmem bank 0)
// MODE 1: pass via global ptr (extra LDG)
//
// Each thread reads the value 8x in a tight loop.
#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    // seed is the 'arg' — passed via cmem bank 0
    int acc = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Read seed 8 times — should hit cmem each time
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int v;
            asm volatile("mov.s32 %0, %1;" : "=r"(v) : "r"(seed));
            acc += v + k;
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (acc == 0xCAFEBABE) C[blockIdx.x] = (float)acc;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=0 (cmem arg) ITERS=%d cy/iter=%.3f cy/op=%.3f\n",
               ITERS, (double)(t1-t0)/(double)ITERS, (double)(t1-t0)/(double)ITERS/8.0);
    }
}
#elif MODE == 1
extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    // 'arg' is in A[0] (global mem), each iter has to LDG
    int acc = 0;
    int* iA = (int*)A;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int v;
            asm volatile("ld.global.b32 %0, [%1];" : "=r"(v) : "l"(iA + (i & 1)));  // i&1 to defeat CSE
            acc += v + k;
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (acc == 0xCAFEBABE) C[blockIdx.x] = (float)acc;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=1 (global LDG arg) ITERS=%d cy/iter=%.3f cy/op=%.3f\n",
               ITERS, (double)(t1-t0)/(double)ITERS, (double)(t1-t0)/(double)ITERS/8.0);
    }
}
#endif
