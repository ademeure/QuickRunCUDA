#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef INNER
#define INNER 32
#endif
#ifndef OUTER
#define OUTER 100
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned a0=0x3c003c00, a1=0x3c003c00, a2=0x3c003c00, a3=0x3c003c00;
    unsigned b0=0x3c003c00, b1=0x3c003c00;
    float c[8][4];
    #pragma unroll
    for (int k=0; k<8; k++) {
        c[k][0] = __int_as_float(tid+k*4+1)*1e-30f;
        c[k][1] = __int_as_float(tid+k*4+2)*1e-30f;
        c[k][2] = __int_as_float(tid+k*4+3)*1e-30f;
        c[k][3] = __int_as_float(tid+k*4+4)*1e-30f;
    }
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            #pragma unroll
            for (int k=0; k<8; k++) {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
            }
        }
    }
    float sum = 0;
    #pragma unroll
    for (int k=0; k<8; k++) sum += c[k][0]+c[k][1]+c[k][2]+c[k][3];
    if (__float_as_int(sum) == seed) C[tid] = sum;
}
