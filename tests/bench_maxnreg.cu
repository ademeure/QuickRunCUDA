// .maxnreg PTX directive effect on SASS register usage.
// Add inline PTX with .maxnreg constraint and see if compiler honors it.

#ifndef MAXREG
#define MAXREG 0
#endif

#if MAXREG > 0
__device__ __forceinline__ void worker(unsigned int& v) {
    asm volatile(".maxnreg " #MAXREG "; nop;");  // doesn't compile - .maxnreg is a kernel attr
}
#endif

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Heavy register usage with many independent chains
    unsigned int v[32];
    #pragma unroll
    for (int k = 0; k < 32; k++) v[k] = (unsigned)(threadIdx.x + k);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            v[k] = v[k] * 31u + (unsigned)i;
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < 32; k++) acc ^= v[k];
    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
