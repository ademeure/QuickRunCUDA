// __half / __half2 atomic add throughput.
// Mode 0: atomicAdd float (baseline)
// Mode 1: atomicAdd __half2 packed
// Mode 2: atomicAdd __half scalar (via h2 trick)
// Mode 3: PTX atom.global.add.noftz.f16x2

#include <cuda_fp16.h>

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float* fa = A;
    __half2* h2a = (__half2*)A;
    __half* ha = (__half*)A;
    __half2 v2 = __float2half2_rn(1.0f);
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 32;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        atomicAdd(fa + (idx & 0x3FF), 1.0f);
#elif MODE == 1
        atomicAdd(h2a + (idx & 0x3FF), v2);
#elif MODE == 2
        atomicAdd(ha + (idx & 0x7FF), __float2half(1.0f));
#elif MODE == 3
        unsigned int v_packed = *((unsigned int*)&v2);
        asm volatile("atom.global.add.noftz.f16x2 _, [%0], %1;"
                     :: "l"(h2a + (idx & 0x3FF)), "r"(v_packed));
#endif
    }
}
