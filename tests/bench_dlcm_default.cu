// Default cache mode for ld/st (ld via dlcm flag).
// Test what nvcc emits for a regular pointer dereference vs explicit hints.

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int* p = (unsigned int*)A;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FF;
    unsigned int v;

#if MODE == 0
    // Plain dereference - what does nvcc default to?
    v = p[idx];
#elif MODE == 1
    // __ldg (read-only)
    v = __ldg(p + idx);
#elif MODE == 2
    // __ldcg (cache global, bypass L1)
    v = __ldcg(p + idx);
#elif MODE == 3
    // __ldca (cache all, default for read+write data)
    v = __ldca(p + idx);
#elif MODE == 4
    // __ldcs (streaming - non-temporal load)
    v = __ldcs(p + idx);
#elif MODE == 5
    // __ldcv (volatile, no cache)
    v = __ldcv(p + idx);
#endif

    p[idx] = v + (unsigned)u2;
}
