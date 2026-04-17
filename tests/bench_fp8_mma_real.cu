// FP8 mma.sync via emulation (F2FP unpack + FP16 HMMA) — defeat DCE properly
// by chaining accumulator through ALL chains and reading inputs fresh per iter.
//
// Key: use loop-carried inputs that change per iteration, not constants.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef INNER
#define INNER 32
#endif
#ifndef OUTER
#define OUTER 100
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Use float regs that are loop-carried so compiler MUST emit each mma
    float c[8][4];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        c[k][0] = __int_as_float(tid + k*4 + 1) * 1e-30f;
        c[k][1] = __int_as_float(tid + k*4 + 2) * 1e-30f;
        c[k][2] = __int_as_float(tid + k*4 + 3) * 1e-30f;
        c[k][3] = __int_as_float(tid + k*4 + 4) * 1e-30f;
    }

#if OP == 0
    // FP8 e4m3 m16n8k32 (emulated). Use chain-dependent inputs: each mma's inputs come from
    // PREVIOUS mma's output via __float_as_int reinterpretation.
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                // Inputs depend on accumulator state from previous iter
                unsigned a0 = __float_as_int(c[k][0]) ^ 0x3c3c3c3c;
                unsigned a1 = __float_as_int(c[k][1]) ^ 0x3c3c3c3c;
                unsigned b0 = __float_as_int(c[k][2]) ^ 0x3c3c3c3c;
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                    : "r"(a0),"r"(a1),"r"(a0),"r"(a1), "r"(b0),"r"(b0));
            }
        }
    }
#elif OP == 1
    // FP8 e5m2 m16n8k32 (emulated)
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        unsigned a0 = __float_as_int(c[0][0]) ^ 0x3c3c3c3c;
        unsigned a1 = __float_as_int(c[1][0]) ^ 0x3c3c3c3c;
        unsigned b0 = __float_as_int(c[2][0]) ^ 0x3c3c3c3c;
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                    : "r"(a0),"r"(a1),"r"(a0),"r"(a1), "r"(b0),"r"(b0));
            }
        }
    }
#endif

    float sum = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) sum += c[k][0]+c[k][1]+c[k][2]+c[k][3];
    if (__float_as_int(sum) == seed) C[tid] = sum;
}
