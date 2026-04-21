// I7: Warp slot allocation — confirm warp_id % 4 → SMSP_id
// Test: read SR_HW_TASK_ID or use a unique SMSP-detection approach
// Approach: have warp 0 do MUFU heavy (slows SMSP); see which OTHER warps are slowed
// If warp 4, 8, 12 are slow → warp_id % 4 mapping (round-robin in groups)
// If warp 1, 2, 3 are slow → consecutive (chunks of 8 per SMSP)

#ifndef MODE
#define MODE 0
#endif

#ifndef NWARPS
#define NWARPS 8
#endif
extern "C" __global__ __launch_bounds__(NWARPS*32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int wid = threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    float a = (float)threadIdx.x * 0.001f + (float)u2 * 1e-9f;
    float ya = (float)(threadIdx.x ^ u2) * 0.002f + 1.0f;
    float za = 0.5f;
    float m = (float)(threadIdx.x ^ u2) * 0.5f + 1.0f;

    unsigned long long t0, t1;
    if (lane == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
            // Warp 0 does heavy MUFU; others do FFMA
            if (wid == 0) {
                asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
            } else {
                a = a*ya + za;
            }
        }
    }

    if (lane == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[blockIdx.x * NWARPS + wid] = t1 - t0;
    }
    if (lane == 0 && a*m == 12345.6f) ((unsigned int*)C)[1024 + wid] = (unsigned)(a + m);
}
