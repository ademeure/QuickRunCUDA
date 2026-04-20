// __forceinline__ vs noinline impact

#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
__device__ __forceinline__ unsigned int op(unsigned int a, unsigned int b) {
    return a * b + a;
}
#else
__device__ __noinline__ unsigned int op(unsigned int a, unsigned int b) {
    return a * b + a;
}
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)threadIdx.x;
    unsigned int b = (unsigned)blockIdx.x + (unsigned)u2;

    for (int i = 0; i < ITERS; i++) {
        v = op(v, b + (unsigned)i);
    }

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
}
