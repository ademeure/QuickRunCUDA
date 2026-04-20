// SHMEM bank conflict / broadcast behavior on B300.
// Modes (each thread reads from a chosen offset):
//   0: All threads read addr[0] (perfect broadcast)
//   1: lane k reads addr[k] (peak, distinct banks)
//   2: lane k reads addr[k % 16] (2-way broadcast: 16 banks shared by 2 lanes)
//   3: lane k reads addr[k % 8]  (4-way broadcast: 8 banks shared by 4 lanes)
//   4: lane k reads addr[k * 2]  (stride-2: 16 active banks, no conflict if x32 wraps)
//   5: lane k reads addr[k * 32] (stride-32: all lanes hit same bank → 32-way conflict)
//   6: lane k reads addr[k * 4]  (stride-4: 8 active banks ÷ 4 = 8 conflicts)
//   7: lane k reads addr[k * 33] (stride-33: bank-skewed, no conflict in theory)

#ifndef ITERS_INNER
#define ITERS_INNER 256
#endif
#ifndef ACCESS_MODE
#define ACCESS_MODE 0
#endif
#ifndef N_DEPS
#define N_DEPS 8
#endif

extern "C" __global__ __launch_bounds__(32, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ unsigned int smem[1024];

    if (threadIdx.x < 32) {
        for (int i = threadIdx.x; i < 1024; i += 32) smem[i] = i ^ 0xDEADBEEFu;
    }
    __syncthreads();

    unsigned int v[N_DEPS];
    #pragma unroll
    for (int k = 0; k < N_DEPS; k++) v[k] = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < N_DEPS; k++) {
            unsigned int lane = threadIdx.x;
            // Use v[k] (chain) to perturb idx unpredictably to compiler.
            // The chain v[k] = v[k] ^ x means each outer iter has a fresh
            // value (depends on previous LDS result), so idx changes too.
            // u2 passed as 0 from runtime - compiler can't constant-fold
            unsigned int perturb = (unsigned int)u2 * v[k];
            unsigned int idx;
#if ACCESS_MODE == 0
            idx = (k * 13) + perturb;
#elif ACCESS_MODE == 1
            idx = lane + (k * 32) + perturb;
#elif ACCESS_MODE == 2
            idx = (lane & 15) + (k * 32) + perturb;
#elif ACCESS_MODE == 3
            idx = (lane & 7)  + (k * 32) + perturb;
#elif ACCESS_MODE == 4
            idx = (lane * 2) + (k * 64) + perturb;
#elif ACCESS_MODE == 5
            idx = (lane * 32) + (k * 1) + perturb;
#elif ACCESS_MODE == 6
            idx = (lane * 4) + (k * 128) + perturb;
#elif ACCESS_MODE == 7
            idx = (lane * 33 + k * 32) + perturb;
#endif
            idx = (idx + ((unsigned)i * (unsigned)u2)) & 0x3FF;
            unsigned int x = smem[idx];
            v[k] = v[k] ^ x;
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_DEPS; k++) acc ^= v[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x] = acc;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total_ld = (unsigned long long)ITERS * (unsigned long long)N_DEPS;
        printf("MODE=%d iters=%d N_DEPS=%d clk=%llu cy/ld=%.3f\n",
               ACCESS_MODE, ITERS, N_DEPS, t1 - t0, (double)(t1-t0)/(double)total_ld);
    }
}
