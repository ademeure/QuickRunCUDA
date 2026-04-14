// Deep REDUX/CREDUX catalog: every data-type + mask variant.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef OP
#define OP 0
#endif
#ifndef MASK
#define MASK 0xFFFFFFFF
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v0 = 0xDEADBEEFu ^ (threadIdx.x * 131);
    unsigned int v1 = 0xCAFEBABEu ^ (threadIdx.x * 97);
    unsigned int v2 = 0x12345678u ^ (threadIdx.x * 71);
    unsigned int v3 = 0xABCDEF01u ^ (threadIdx.x * 53);
    unsigned int v4 = 0xFEEDFACEu ^ (threadIdx.x * 41);
    unsigned int v5 = 0xBADC0DE1u ^ (threadIdx.x * 29);
    unsigned int v6 = 0xDEAFBEEFu ^ (threadIdx.x * 17);
    unsigned int v7 = 0x00000001u ^ (threadIdx.x * 11);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // u32 min, full mask
                asm volatile("redux.sync.min.u32 %0, %0, %1;" : "+r"(v0) : "n"(MASK));
                asm volatile("redux.sync.min.u32 %0, %0, %1;" : "+r"(v1) : "n"(MASK));
                asm volatile("redux.sync.min.u32 %0, %0, %1;" : "+r"(v2) : "n"(MASK));
                asm volatile("redux.sync.min.u32 %0, %0, %1;" : "+r"(v3) : "n"(MASK));
                asm volatile("redux.sync.min.u32 %0, %0, %1;" : "+r"(v4) : "n"(MASK));
                asm volatile("redux.sync.min.u32 %0, %0, %1;" : "+r"(v5) : "n"(MASK));
                asm volatile("redux.sync.min.u32 %0, %0, %1;" : "+r"(v6) : "n"(MASK));
                asm volatile("redux.sync.min.u32 %0, %0, %1;" : "+r"(v7) : "n"(MASK));
#elif OP == 1  // s32 min
                asm volatile("redux.sync.min.s32 %0, %0, %1;" : "+r"(v0) : "n"(MASK));
                asm volatile("redux.sync.min.s32 %0, %0, %1;" : "+r"(v1) : "n"(MASK));
                asm volatile("redux.sync.min.s32 %0, %0, %1;" : "+r"(v2) : "n"(MASK));
                asm volatile("redux.sync.min.s32 %0, %0, %1;" : "+r"(v3) : "n"(MASK));
                asm volatile("redux.sync.min.s32 %0, %0, %1;" : "+r"(v4) : "n"(MASK));
                asm volatile("redux.sync.min.s32 %0, %0, %1;" : "+r"(v5) : "n"(MASK));
                asm volatile("redux.sync.min.s32 %0, %0, %1;" : "+r"(v6) : "n"(MASK));
                asm volatile("redux.sync.min.s32 %0, %0, %1;" : "+r"(v7) : "n"(MASK));
#elif OP == 2  // f32 min (may not exist)
                {float f0=__int_as_float(v0),f1=__int_as_float(v1),f2=__int_as_float(v2),f3=__int_as_float(v3);
                 float f4=__int_as_float(v4),f5=__int_as_float(v5),f6=__int_as_float(v6),f7=__int_as_float(v7);
                 asm volatile("redux.sync.min.f32 %0, %0, %1;" : "+f"(f0) : "n"(MASK));
                 asm volatile("redux.sync.min.f32 %0, %0, %1;" : "+f"(f1) : "n"(MASK));
                 asm volatile("redux.sync.min.f32 %0, %0, %1;" : "+f"(f2) : "n"(MASK));
                 asm volatile("redux.sync.min.f32 %0, %0, %1;" : "+f"(f3) : "n"(MASK));
                 asm volatile("redux.sync.min.f32 %0, %0, %1;" : "+f"(f4) : "n"(MASK));
                 asm volatile("redux.sync.min.f32 %0, %0, %1;" : "+f"(f5) : "n"(MASK));
                 asm volatile("redux.sync.min.f32 %0, %0, %1;" : "+f"(f6) : "n"(MASK));
                 asm volatile("redux.sync.min.f32 %0, %0, %1;" : "+f"(f7) : "n"(MASK));
                 v0=__float_as_int(f0);v1=__float_as_int(f1);v2=__float_as_int(f2);v3=__float_as_int(f3);
                 v4=__float_as_int(f4);v5=__float_as_int(f5);v6=__float_as_int(f6);v7=__float_as_int(f7);}
#elif OP == 3  // f32 min.NaN
                {float f0=__int_as_float(v0),f1=__int_as_float(v1),f2=__int_as_float(v2),f3=__int_as_float(v3);
                 float f4=__int_as_float(v4),f5=__int_as_float(v5),f6=__int_as_float(v6),f7=__int_as_float(v7);
                 asm volatile("redux.sync.min.NaN.f32 %0, %0, %1;" : "+f"(f0) : "n"(MASK));
                 asm volatile("redux.sync.min.NaN.f32 %0, %0, %1;" : "+f"(f1) : "n"(MASK));
                 asm volatile("redux.sync.min.NaN.f32 %0, %0, %1;" : "+f"(f2) : "n"(MASK));
                 asm volatile("redux.sync.min.NaN.f32 %0, %0, %1;" : "+f"(f3) : "n"(MASK));
                 asm volatile("redux.sync.min.NaN.f32 %0, %0, %1;" : "+f"(f4) : "n"(MASK));
                 asm volatile("redux.sync.min.NaN.f32 %0, %0, %1;" : "+f"(f5) : "n"(MASK));
                 asm volatile("redux.sync.min.NaN.f32 %0, %0, %1;" : "+f"(f6) : "n"(MASK));
                 asm volatile("redux.sync.min.NaN.f32 %0, %0, %1;" : "+f"(f7) : "n"(MASK));
                 v0=__float_as_int(f0);v1=__float_as_int(f1);v2=__float_as_int(f2);v3=__float_as_int(f3);
                 v4=__float_as_int(f4);v5=__float_as_int(f5);v6=__float_as_int(f6);v7=__float_as_int(f7);}
#elif OP == 4  // redux.sync.add.f32 (if exists)
                {float f0=__int_as_float(v0),f1=__int_as_float(v1),f2=__int_as_float(v2),f3=__int_as_float(v3);
                 float f4=__int_as_float(v4),f5=__int_as_float(v5),f6=__int_as_float(v6),f7=__int_as_float(v7);
                 asm volatile("redux.sync.add.f32 %0, %0, %1;" : "+f"(f0) : "n"(MASK));
                 asm volatile("redux.sync.add.f32 %0, %0, %1;" : "+f"(f1) : "n"(MASK));
                 asm volatile("redux.sync.add.f32 %0, %0, %1;" : "+f"(f2) : "n"(MASK));
                 asm volatile("redux.sync.add.f32 %0, %0, %1;" : "+f"(f3) : "n"(MASK));
                 asm volatile("redux.sync.add.f32 %0, %0, %1;" : "+f"(f4) : "n"(MASK));
                 asm volatile("redux.sync.add.f32 %0, %0, %1;" : "+f"(f5) : "n"(MASK));
                 asm volatile("redux.sync.add.f32 %0, %0, %1;" : "+f"(f6) : "n"(MASK));
                 asm volatile("redux.sync.add.f32 %0, %0, %1;" : "+f"(f7) : "n"(MASK));
                 v0=__float_as_int(f0);v1=__float_as_int(f1);v2=__float_as_int(f2);v3=__float_as_int(f3);
                 v4=__float_as_int(f4);v5=__float_as_int(f5);v6=__float_as_int(f6);v7=__float_as_int(f7);}
#elif OP == 5  // u32 add (reference)
                asm volatile("redux.sync.add.u32 %0, %0, %1;" : "+r"(v0) : "n"(MASK));
                asm volatile("redux.sync.add.u32 %0, %0, %1;" : "+r"(v1) : "n"(MASK));
                asm volatile("redux.sync.add.u32 %0, %0, %1;" : "+r"(v2) : "n"(MASK));
                asm volatile("redux.sync.add.u32 %0, %0, %1;" : "+r"(v3) : "n"(MASK));
                asm volatile("redux.sync.add.u32 %0, %0, %1;" : "+r"(v4) : "n"(MASK));
                asm volatile("redux.sync.add.u32 %0, %0, %1;" : "+r"(v5) : "n"(MASK));
                asm volatile("redux.sync.add.u32 %0, %0, %1;" : "+r"(v6) : "n"(MASK));
                asm volatile("redux.sync.add.u32 %0, %0, %1;" : "+r"(v7) : "n"(MASK));
#endif
        }
    }
    unsigned int acc = v0^v1^v2^v3^v4^v5^v6^v7;
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
