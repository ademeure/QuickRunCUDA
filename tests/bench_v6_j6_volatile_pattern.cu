// V6 J6: volatile reads in tight loop — does HW recognize patterns?
// MODE 0: 8 normal global reads (compiler caches in reg, only 1 actual load)
// MODE 1: 8 volatile global reads (each must hit memory; verify SASS)
// MODE 2: 8 volatile reads spread across cache lines (no reuse)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float acc = 0.0f;
    int idx = threadIdx.x;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Normal reads — compiler may CSE
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            acc += A[idx];
        }
#elif MODE == 1
        // Volatile reads from same address — each MUST hit memory
        volatile float* va = (volatile float*)A;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            acc += va[idx];
        }
#elif MODE == 2
        // Volatile reads from different cache lines (32B apart = 8 floats)
        volatile float* va = (volatile float*)A;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            acc += va[idx + k * 32];
        }
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (acc == 1.234567e-30f) C[blockIdx.x] = acc;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d cy/iter=%.3f\n",
               MODE, ITERS, (double)(t1-t0)/(double)ITERS);
    }
}
