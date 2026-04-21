// A2: Warp scheduler policy under contention
// Each warp records its own clock64 start/end and per-iter timestamps
// Examine fairness: are all warps getting equal cycles, or some starved?
//
// MODE 0: All warps do FFMA (uniform work, expect fair)
// MODE 1: All warps do LDS chain (uniform memory)
// MODE 2: Half warps FFMA, half LDS (mixed pipe — scheduler may favor one)
// MODE 3: One warp does HEAVY chained MUFU, rest do FFMA
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

    __shared__ unsigned int smem[1024];
    if (lane == 0) {
        for (int i = 0; i < 32; i++) smem[wid*32 + i] = i + (unsigned)u2;
    }
    __syncthreads();

    unsigned int base_addr = __cvta_generic_to_shared(smem);
    float a = (float)threadIdx.x * 0.001f + (float)u2 * 1e-9f;
    float ya = (float)(threadIdx.x ^ u2) * 0.002f + 1.0f;
    float za = 0.5f;
    float m = (float)(threadIdx.x ^ u2) * 0.5f + 1.0f;
    unsigned int v = (unsigned)(threadIdx.x ^ u2);

    unsigned long long t0, t1;
    if (lane == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            // All warps: FFMA
            a = a*ya + za;
#elif MODE == 1
            // All warps: LDS chain (with own slot)
            unsigned int x;
            unsigned int off = ((v & 31) + wid*32) * 4;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(base_addr + off));
            v = x;
#elif MODE == 2
            // Mixed: even warps FFMA, odd warps LDS
            if ((wid & 1) == 0) {
                a = a*ya + za;
            } else {
                unsigned int x;
                unsigned int off = ((v & 31) + wid*32) * 4;
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(base_addr + off));
                v = x;
            }
#elif MODE == 3
            // Warp 0: heavy MUFU (rsqrt 18 cy/op); others: light FFMA
            if (wid == 0) {
                asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
            } else {
                a = a*ya + za;
            }
#endif
        }
    }

    if (lane == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        // Each warp logs its time to C[wid] (per-block)
        ((unsigned long long*)C)[blockIdx.x * NWARPS + wid] = t1 - t0;
    }

    if (lane == 0 && a*v*m == 12345.6789f) ((unsigned int*)C)[1024 + wid] = (unsigned)a + v + (unsigned)m;
}
