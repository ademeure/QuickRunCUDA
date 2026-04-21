// V10: LDG cache hint variants (.ca / .cg / .cs / .lu)
#ifndef HINT
#define HINT 0  // 0=ca (default), 1=cg (L2 only), 2=cs (streaming), 3=lu (last use)
#endif
#ifndef K_INNER
#define K_INNER 64
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int THREADS = gridDim.x * blockDim.x;
    long long T = THREADS;

    float4 acc = make_float4(0, 0, 0, 0);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 8
        for (int k = 0; k < K_INNER; k++) {
            long long idx = (long long)i * T * K_INNER + (long long)k * T + (long long)gtid;
            float4 v;
#if HINT == 0
            asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];"
                         : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(A + idx));
#elif HINT == 1
            asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
                         : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(A + idx));
#elif HINT == 2
            asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];"
                         : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(A + idx));
#elif HINT == 3
            asm volatile("ld.global.lu.v4.f32 {%0,%1,%2,%3}, [%4];"
                         : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(A + idx));
#endif
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
    }

    if (acc.x == 1.234567e-30f) C[gtid] = acc;
}
