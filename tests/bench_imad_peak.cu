#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#ifndef INNER
#define INNER 128
#endif
#ifndef OUTER
#define OUTER 100
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned v0=tid+1,v1=tid+2,v2=tid+3,v3=tid+4;
    unsigned v4=tid+5,v5=tid+6,v6=tid+7,v7=tid+8;
    unsigned y = (unsigned)seed | 1u;  // runtime, ensures odd

    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            // IMAD chain v = v*y + v  (each chain dependent)
            v0 = v0 * y + v0; v1 = v1 * y + v1;
            v2 = v2 * y + v2; v3 = v3 * y + v3;
            v4 = v4 * y + v4; v5 = v5 * y + v5;
            v6 = v6 * y + v6; v7 = v7 * y + v7;
        }
    }
    unsigned sum = v0+v1+v2+v3+v4+v5+v6+v7;
    if (sum == (unsigned)seed) C[tid] = sum;
}
