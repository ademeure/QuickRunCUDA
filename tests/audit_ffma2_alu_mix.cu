// §22 audit: FFMA2 + LOP3 dual-issue throughput.
//
// Goal: test whether packed FFMA2 (which uses ONE dispatch slot for 2 FFMA = 4 FLOPS)
// can be co-issued with LOP3 (ALU pipe) to give MORE useful ops/SM/cy than either solo.
//
// Modes (selected by -H "#define N_FFMA2 X" and -H "#define N_LOP3 Y"):
//   - N_FFMA2 only         → solo FFMA2 throughput
//   - N_LOP3 only          → solo LOP3 throughput
//   - both nonzero         → mixed at ratio N_FFMA2:N_LOP3 per inner slot
//
// Anti-DCE: store XOR-reduced accumulator under impossible predicate.
// Anti-LICM: register init from threadIdx.x.

#ifndef N_FFMA2
#define N_FFMA2 0
#endif
#ifndef N_LOP3
#define N_LOP3 0
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int unused1, int unused2) {
    int tid = threadIdx.x;

#if N_FFMA2 > 0
    // Each chain holds a packed pair of fp32 in one u64.
    unsigned long long f[N_FFMA2];
    #pragma unroll
    for (int k = 0; k < N_FFMA2; k++) {
        unsigned int ulo = __float_as_int(1.0001f + 0.0001f*(tid + k*23));
        unsigned int uhi = __float_as_int(1.0002f + 0.0001f*(tid + k*29));
        f[k] = ((unsigned long long)uhi << 32) | ulo;
    }
    // Two fp32x2 constants packed in u64.
    unsigned int c1_u = __float_as_int(1.000001f);
    unsigned int c0_u = __float_as_int(0.9999f);
    unsigned long long c1 = ((unsigned long long)c1_u << 32) | c1_u;
    unsigned long long c0 = ((unsigned long long)c0_u << 32) | c0_u;
#endif

#if N_LOP3 > 0
    unsigned int u[N_LOP3];
    #pragma unroll
    for (int k = 0; k < N_LOP3; k++) {
        u[k] = (unsigned int)(tid * 7u + k * 13u + 0xab);
    }
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // Issue all FFMA2s for this slot.
#if N_FFMA2 > 0
            #pragma unroll
            for (int k = 0; k < N_FFMA2; k++) {
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;"
                             : "+l"(f[k]) : "l"(c1), "l"(c0));
            }
#endif
            // Issue all LOP3s for this slot.
#if N_LOP3 > 0
            #pragma unroll
            for (int k = 0; k < N_LOP3; k++) {
                asm volatile("lop3.b32 %0, %0, 0xa5a5a5a5, 0x12345678, 0x96;"
                             : "+r"(u[k]));
            }
#endif
        }
    }

    // Anti-DCE: gather all accumulators into one scalar, store under
    // impossible predicate (BLOCK_SIZE..tid never reached).
    unsigned long long acc = 0;
#if N_FFMA2 > 0
    #pragma unroll
    for (int k = 0; k < N_FFMA2; k++) acc ^= f[k];
#endif
#if N_LOP3 > 0
    #pragma unroll
    for (int k = 0; k < N_LOP3; k++) acc ^= (unsigned long long)u[k];
#endif
    if (tid >= blockDim.x) {
        ((unsigned long long*)C)[blockIdx.x * blockDim.x + tid] = acc;
    }
}
