// Test 8: write to a fresh dst, read all 3 sources from same OTHER reg
// fma %0, %1, %1, %1   -- destination is different than the all-three-sources reg
// This isolates write-port vs read-port: read 1 reg 3 ways, write to a different reg
// Then chain via dst -> next iter's src
#ifndef N_INNER
#define N_INNER 1024
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (threadIdx.x >= 32) return;

    // Two ping-pong regs; src is one, dst is the other; swap each iter (manual unroll-2)
    float v0 = (float)threadIdx.x * 0.001f + 1.0001f;
    float v1 = 0.0f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll (N_INNER/2)
        for (int j = 0; j < (N_INNER/2); j++) {
            // v1 = v0*v0 + v0   -> v0 read 3 ways, written to v1
            asm volatile("fma.rn.f32 %0, %1, %1, %1;" : "=f"(v1) : "f"(v0));
            // v0 = v1*v1 + v1
            asm volatile("fma.rn.f32 %0, %1, %1, %1;" : "=f"(v0) : "f"(v1));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float v = v0 + v1;
    if ((int)v == seed) C[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
        printf("v8_outdiff_inputs total=%llu clk=%llu cy/op=%.4f\n",
               total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
