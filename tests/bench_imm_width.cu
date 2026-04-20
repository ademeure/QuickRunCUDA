// Maximum immediate width for FFMA/IMAD/IMNMX/LOP3.
// Test if compiler embeds immediate in the inst or falls back to ULDC.
// Vary the immediate value across small (4-bit), medium (16-bit), large (24-bit), full 32-bit.

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef IMM_VAL
#define IMM_VAL 0x7
#endif
#ifndef OP_MODE
#define OP_MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) v[k] = (unsigned)(threadIdx.x + k);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if OP_MODE == 0
                // IMAD with imm
                v[k] = v[k] * IMM_VAL + IMM_VAL;
#elif OP_MODE == 1
                // LOP3 / AND with imm
                v[k] = v[k] & IMM_VAL;
                v[k] = v[k] | (IMM_VAL << 4);
#elif OP_MODE == 2
                // FFMA with float imm (cast)
                float fv = __uint_as_float(v[k]);
                fv = fv * (float)(IMM_VAL) + (float)(IMM_VAL);
                v[k] = __float_as_uint(fv);
#elif OP_MODE == 3
                // IMNMX with imm
                v[k] = (v[k] < IMM_VAL) ? v[k] : IMM_VAL;
#endif
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
