// All mma.sync precision peak audit. 8 chains of independent mma.sync.
// OP=0 : f16->f32 m16n8k16 (already audited 577)
// OP=1 : bf16->f32 m16n8k16
// OP=2 : tf32->f32 m16n8k8
// OP=3 : f8e4m3->f32 m16n8k32
// OP=4 : f8e5m2->f32 m16n8k32
// OP=5 : s8->s32 m16n8k32 (INT8)

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
    unsigned a0=0x3c003c00, a1=0x3c003c00, a2=0x3c003c00, a3=0x3c003c00;  // FP16=1.0 packed
    unsigned b0=0x3c003c00, b1=0x3c003c00;
    unsigned bf_a0=0x3f803f80, bf_a1=0x3f803f80, bf_a2=0x3f803f80, bf_a3=0x3f803f80; // BF16=1.0 packed
    unsigned bf_b0=0x3f803f80, bf_b1=0x3f803f80;
    unsigned tf_a0=0x3f8003ff, tf_a1=0x3f8003ff;  // TF32 1.0 nominal packed
    unsigned tf_b0=0x3f8003ff;
    unsigned f8_a0=0x3c3c3c3c, f8_a1=0x3c3c3c3c;  // E4M3=1.0 packed
    unsigned f8_b0=0x3c3c3c3c;
    unsigned s8_a0=0x01010101, s8_a1=0x01010101;
    unsigned s8_b0=0x01010101;

    float c[8][4];
    int   ic[8][4];
    #pragma unroll
    for (int k=0; k<8; k++) {
        c[k][0] = __int_as_float(tid+k*4+1)*1e-30f;
        c[k][1] = __int_as_float(tid+k*4+2)*1e-30f;
        c[k][2] = __int_as_float(tid+k*4+3)*1e-30f;
        c[k][3] = __int_as_float(tid+k*4+4)*1e-30f;
        ic[k][0] = (int)(tid+k); ic[k][1]=ic[k][2]=ic[k][3]=0;
    }

    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            #pragma unroll
            for (int k=0; k<8; k++) {
#if OP == 0
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#elif OP == 1
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                    : "r"(bf_a0),"r"(bf_a1),"r"(bf_a2),"r"(bf_a3), "r"(bf_b0),"r"(bf_b1));
#elif OP == 2
                // TF32 m16n8k8: A=4 regs, B=2 regs
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                    : "r"(tf_a0),"r"(tf_a1),"r"(tf_a0),"r"(tf_a1), "r"(tf_b0),"r"(tf_b0));
#elif OP == 3
                // FP8 m16n8k32: A=4 regs, B=2 regs
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                    : "r"(f8_a0),"r"(f8_a1),"r"(f8_a0),"r"(f8_a1), "r"(f8_b0),"r"(f8_b0));
#elif OP == 4
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                    : "r"(f8_a0),"r"(f8_a1),"r"(f8_a0),"r"(f8_a1), "r"(f8_b0),"r"(f8_b0));
#elif OP == 5
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+r"(ic[k][0]),"+r"(ic[k][1]),"+r"(ic[k][2]),"+r"(ic[k][3])
                    : "r"(s8_a0),"r"(s8_a1),"r"(s8_a0),"r"(s8_a1), "r"(s8_b0),"r"(s8_b0));
#endif
            }
        }
    }
    float sum = 0;
    int isum = 0;
    #pragma unroll
    for (int k=0; k<8; k++) {
        sum += c[k][0]+c[k][1]+c[k][2]+c[k][3];
        isum += ic[k][0]+ic[k][1]+ic[k][2]+ic[k][3];
    }
    if (__float_as_int(sum) == seed) C[tid] = sum;
    if (isum == seed) C[tid+1024] = (float)isum;
}
