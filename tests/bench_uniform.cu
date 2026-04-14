// Force uniform-datapath emission.
// Trick: warp-uniform values (same across lanes) let compiler use URx regs.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // blockIdx.x and seed are uniform across the warp — compiler may place in URx
    unsigned int u0 = blockIdx.x;
    unsigned int u1 = blockIdx.x + 17;
    unsigned int u2_ = (unsigned)seed;
    unsigned int u3 = (unsigned)seed * 3;
    float f0 = __int_as_float(u0);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // uniform iadd chain — should emit UIADD3
            u0 = u0 * 3 + u1;
            u1 = u1 + 17;
            u2_ = u2_ * 5 + u3;
            u3 = u3 + 23;
#elif OP == 1  // uniform FMUL via blockIdx-dependent
            float f = f0;
            f = f * 1.000001f + 0.0001f;
            f0 = f;
#elif OP == 2  // uniform LOP3
            u0 = u0 ^ u1;
            u1 = u1 ^ u2_;
            u2_ = u2_ ^ u3;
            u3 = u3 ^ u0;
#elif OP == 3  // mixed per-lane with uniform arg (forces UMOV broadcast)
            unsigned int x = threadIdx.x * u0;
            x = x ^ u1;
            u0 = __shfl_sync(0xFFFFFFFF, x, 0);  // bring back to uniform
#endif
        }
    }
    if (u0 == (unsigned)seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = u0 + __float_as_int(f0);
}
