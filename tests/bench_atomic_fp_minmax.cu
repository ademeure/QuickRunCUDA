// atomicMin/Max FP throughput on B300
// Mode 0: atomicAdd float (baseline)
// Mode 1: atomicMin float (new in CC 9.0+ via __float_as_int trick or native?)
// Mode 2: atomicMax float
// Mode 3: atomicAdd __half (FP16)
// Mode 4: atomicCAS float-based min (manual)

#include <cuda_fp16.h>

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float* fa = A;
    int* ia = (int*)A;
    __half* ha = (__half*)A;

    // Each thread targets a different cache line to defeat contention
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 32;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        atomicAdd(fa + (idx & 0x3FF), 1.0f);
#elif MODE == 1
        atomicMin(ia + (idx & 0x3FF), i);  // int min as proxy
#elif MODE == 2
        atomicMax(ia + (idx & 0x3FF), i);
#elif MODE == 3
        atomicAdd(ha + (idx & 0x7FF), __float2half(1.0f));
#elif MODE == 4
        // float min via atomicCAS
        int* p = ia + (idx & 0x3FF);
        int old = *p;
        int new_val = __float_as_int(fminf(__int_as_float(old), (float)i));
        atomicCAS(p, old, new_val);
#endif
    }
}
