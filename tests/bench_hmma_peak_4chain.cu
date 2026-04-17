// HMMA peak audit: same methodology as FFMA. Instead of FFMA chains, use
// mma.sync.m16n8k16 in independent chains.
// HMMA m16n8k16 = 1024 FP16 ops × 16/8/16 = 16*8*16*2 = 4096 FLOPs per inst (per warp)
// Per warp: each mma.m16n8k16 = 4096 FLOPs (m*n*k*2_FLOPs_per_FMA)

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
    unsigned lane = tid & 31;

    // Per-warp 4 independent HMMA chains. Each warp has 4 chains in d-registers.
    // FP16 inputs are packed in u32 registers: a/b in 16-bit each, packed pairs.
    unsigned a0=0x3c003c00, a1=0x3c003c00, a2=0x3c003c00, a3=0x3c003c00;  // 1.0 in fp16
    unsigned b0=0x3c003c00, b1=0x3c003c00;
    float c0_0=__int_as_float(tid+1)*1e-30f, c0_1=__int_as_float(tid+2)*1e-30f;
    float c0_2=__int_as_float(tid+3)*1e-30f, c0_3=__int_as_float(tid+4)*1e-30f;
    float c1_0=__int_as_float(tid+5)*1e-30f, c1_1=__int_as_float(tid+6)*1e-30f;
    float c1_2=__int_as_float(tid+7)*1e-30f, c1_3=__int_as_float(tid+8)*1e-30f;
    float c2_0=__int_as_float(tid+9)*1e-30f, c2_1=__int_as_float(tid+10)*1e-30f;
    float c2_2=__int_as_float(tid+11)*1e-30f, c2_3=__int_as_float(tid+12)*1e-30f;
    float c3_0=__int_as_float(tid+13)*1e-30f, c3_1=__int_as_float(tid+14)*1e-30f;
    float c3_2=__int_as_float(tid+15)*1e-30f, c3_3=__int_as_float(tid+16)*1e-30f;

    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            // 4 HMMA chains per inner iter
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c0_0),"+f"(c0_1),"+f"(c0_2),"+f"(c0_3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c1_0),"+f"(c1_1),"+f"(c1_2),"+f"(c1_3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c2_0),"+f"(c2_1),"+f"(c2_2),"+f"(c2_3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c3_0),"+f"(c3_1),"+f"(c3_2),"+f"(c3_3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
        }
    }
    float sum = c0_0+c0_1+c0_2+c0_3 + c1_0+c1_1+c1_2+c1_3
              + c2_0+c2_1+c2_2+c2_3 + c3_0+c3_1+c3_2+c3_3;
    if (__float_as_int(sum) == seed) C[tid] = sum;
}
