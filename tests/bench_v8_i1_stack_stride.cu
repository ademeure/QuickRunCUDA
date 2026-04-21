// V8 I1: HBM stack interleave granularity (v2 — more work per thread)
//
// Each thread does K_INNER loads per outer iter, stride = STRIDE_FL4 float4 apart.
// Different outer iters advance base so successive loads don't hit L2.
//
// MODE 0: STRIDE_FL4 = 192 = 3072 B = 12 lines. Adjacent threads' lines alias
//         to same stack IF hardware interleave granularity is 256 B (= 1 line).
//         All 37888 threads would hit ~1 stack → bw ~600 GB/s.
//
// MODE 1: STRIDE_FL4 = 1 (natural stripe). 37888 threads span all stacks → 7+ TB/s.
// MODE 2: STRIDE_FL4 = 16 (= 1 cache line per thread). Each thread owns a line.
// MODE 3: STRIDE_FL4 = 32 (= 2 lines per thread).

#ifndef MODE
#define MODE 1
#endif
#ifndef K_INNER
#define K_INNER 64
#endif

#if MODE == 0
#define STRIDE_FL4 192
#elif MODE == 1
#define STRIDE_FL4 1
#elif MODE == 2
#define STRIDE_FL4 16
#elif MODE == 3
#define STRIDE_FL4 32
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int THREADS = gridDim.x * blockDim.x;
    long long T = THREADS;
    long long S = STRIDE_FL4;

    float4 acc0 = make_float4(0,0,0,0);
    float4 acc1 = make_float4(0,0,0,0);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Each outer iter operates on a fresh THREADS × K_INNER × STRIDE_FL4 region
        long long base_iter = (long long)i * T * (long long)K_INNER * S;

        #pragma unroll 8
        for (int k = 0; k < K_INNER; k++) {
            long long idx = base_iter + (long long)k * T * S + (long long)gtid * S;
            float4 v = A[idx];
            if (k & 1) { acc1.x += v.x; acc1.y += v.y; acc1.z += v.z; acc1.w += v.w; }
            else       { acc0.x += v.x; acc0.y += v.y; acc0.z += v.z; acc0.w += v.w; }
        }
    }

    if (acc0.x + acc1.x == 1.234567e-30f) C[gtid] = acc0;
}
