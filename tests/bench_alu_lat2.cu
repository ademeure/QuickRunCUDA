// ALU latency — pragma unroll fully, measure pure chain cost.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef N
#define N 1024
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = threadIdx.x;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

#if OP == 0
    float a = __int_as_float(tid + seed);
    float b = 1.0001f;
    #pragma unroll
    for (int i = 0; i < N; i++) a = a * b + a;  // FFMA chain
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = __float_as_int(a); }
#elif OP == 1
    float a = __int_as_float(tid + seed);
    #pragma unroll
    for (int i = 0; i < N; i++) a = a + 1.0001f;  // FADD chain
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = __float_as_int(a); }
#elif OP == 2
    float a = __int_as_float(tid + seed);
    #pragma unroll
    for (int i = 0; i < N; i++) a = a * 1.0001f;  // FMUL chain
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = __float_as_int(a); }
#elif OP == 3
    unsigned a = tid + seed;
    #pragma unroll
    for (int i = 0; i < N; i++) asm volatile("iadd3 %0, %0, 1, %0;" : "+r"(a));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = a; }
#elif OP == 4
    unsigned a = tid + seed;
    unsigned y = (unsigned)seed | 1u;
    #pragma unroll
    for (int i = 0; i < N; i++) a = a * y + a;  // IMAD chain
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = a; }
#elif OP == 5
    unsigned a = tid + seed;
    #pragma unroll
    for (int i = 0; i < N; i++) a = a ^ 0x12345678u;  // LOP3 chain
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = a; }
#elif OP == 6
    double a = (double)(tid + seed);
    double b = 1.000001;
    #pragma unroll
    for (int i = 0; i < N; i++) a = a * b + a;  // DFMA chain
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = (unsigned)__double_as_longlong(a); }
#elif OP == 7
    // HFMA (fp16) chain
    __half a = __float2half((float)(tid + seed));
    __half b = __float2half(1.01f);
    #pragma unroll
    for (int i = 0; i < N; i++) a = __hfma(a, b, a);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = (unsigned)__half_as_ushort(a); }
#elif OP == 8
    // BFMA (bf16) chain
    __nv_bfloat16 a = __float2bfloat16((float)(tid + seed));
    __nv_bfloat16 b = __float2bfloat16(1.01f);
    #pragma unroll
    for (int i = 0; i < N; i++) a = __hfma(a, b, a);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = 0; }
#elif OP == 9
    // HFMA2 (fp16x2) packed
    __half2 a = __float2half2_rn((float)(tid + seed));
    __half2 b = __float2half2_rn(1.01f);
    #pragma unroll
    for (int i = 0; i < N; i++) a = __hfma2(a, b, a);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = 0; }
#elif OP == 10
    // FFMA2 packed float2
    float2 a = {__int_as_float(tid+1), __int_as_float(tid+2)};
    float2 b = {1.01f, 1.01f};
    #pragma unroll
    for (int i = 0; i < N; i++) {
        asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&a) : "l"(*(unsigned long long*)&b), "l"(*(unsigned long long*)&a));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = 0; }
#endif
}
