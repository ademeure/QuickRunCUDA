// V8: HBM write peak via plain STG.E.128 stores
// Theoretical: 7.2-7.5 TB/s HBM3E read; writes should be similar.
// Pattern: each thread writes float4 to sequential locations, cross-grid stride.

#ifndef K_INNER
#define K_INNER 64
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int THREADS = gridDim.x * blockDim.x;
    long long T = THREADS;

    float4 val = make_float4(
        (float)(gtid + seed),
        (float)(gtid - seed),
        (float)(gtid * 37),
        (float)(gtid ^ seed)
    );

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        long long base = (long long)i * T * K_INNER;
        #pragma unroll 8
        for (int k = 0; k < K_INNER; k++) {
            long long idx = base + (long long)k * T + (long long)gtid;
            A[idx] = val;
            // Vary val slightly each iter to defeat any write-combining
            val.x += 1.0f;
        }
    }

    // Anti-DCE: write last val to C
    if (val.x == 1.234567e-30f) C[gtid] = val;
}
