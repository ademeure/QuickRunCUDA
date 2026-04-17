// Rigorous ALU latency AND throughput measurement.
// LATENCY: 1 chain, tight data dep — each op waits for previous.
// THROUGHPUT: 8 indep chains — fully parallel, measures dispatch cap.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef N
#define N 1024
#endif
#ifndef OP
#define OP 0
#endif
#ifndef MODE
#define MODE 0  // 0=latency (1 chain), 1=throughput (8 chains)
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = threadIdx.x;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

#if OP == 0 && MODE == 0
    // FFMA LATENCY: single chain
    float a = __int_as_float(tid + seed);
    float b = 1.0001f;
    #pragma unroll
    for (int i = 0; i < N; i++) a = a * b + a;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = __float_as_int(a); }

#elif OP == 0 && MODE == 1
    // FFMA THROUGHPUT: 8 indep chains
    float a0 = __int_as_float(tid + seed + 1);
    float a1 = __int_as_float(tid + seed + 2);
    float a2 = __int_as_float(tid + seed + 3);
    float a3 = __int_as_float(tid + seed + 4);
    float a4 = __int_as_float(tid + seed + 5);
    float a5 = __int_as_float(tid + seed + 6);
    float a6 = __int_as_float(tid + seed + 7);
    float a7 = __int_as_float(tid + seed + 8);
    float b = 1.0001f;
    #pragma unroll
    for (int i = 0; i < N/8; i++) {
        a0 = a0*b + a0; a1 = a1*b + a1; a2 = a2*b + a2; a3 = a3*b + a3;
        a4 = a4*b + a4; a5 = a5*b + a5; a6 = a6*b + a6; a7 = a7*b + a7;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    float s = a0+a1+a2+a3+a4+a5+a6+a7;
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = __float_as_int(s); }

#elif OP == 1 && MODE == 0
    // FADD LATENCY: tight dep, compiler can't fold if b comes from volatile
    float a = __int_as_float(tid + seed);
    float b_vol = __int_as_float(seed) * 1e-9f;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        asm volatile("add.rn.ftz.f32 %0, %0, %1;" : "+f"(a) : "f"(b_vol));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = __float_as_int(a); }

#elif OP == 1 && MODE == 1
    // FADD THROUGHPUT: 8 chains
    float a0=tid+1.f,a1=tid+2.f,a2=tid+3.f,a3=tid+4.f;
    float a4=tid+5.f,a5=tid+6.f,a6=tid+7.f,a7=tid+8.f;
    float b_vol = __int_as_float(seed) * 1e-9f;
    #pragma unroll
    for (int i = 0; i < N/8; i++) {
        asm volatile("add.rn.ftz.f32 %0, %0, %1;" : "+f"(a0) : "f"(b_vol));
        asm volatile("add.rn.ftz.f32 %0, %0, %1;" : "+f"(a1) : "f"(b_vol));
        asm volatile("add.rn.ftz.f32 %0, %0, %1;" : "+f"(a2) : "f"(b_vol));
        asm volatile("add.rn.ftz.f32 %0, %0, %1;" : "+f"(a3) : "f"(b_vol));
        asm volatile("add.rn.ftz.f32 %0, %0, %1;" : "+f"(a4) : "f"(b_vol));
        asm volatile("add.rn.ftz.f32 %0, %0, %1;" : "+f"(a5) : "f"(b_vol));
        asm volatile("add.rn.ftz.f32 %0, %0, %1;" : "+f"(a6) : "f"(b_vol));
        asm volatile("add.rn.ftz.f32 %0, %0, %1;" : "+f"(a7) : "f"(b_vol));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    float s = a0+a1+a2+a3+a4+a5+a6+a7;
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = __float_as_int(s); }

#elif OP == 2 && MODE == 0
    // IADD3 LATENCY
    unsigned a = tid + seed;
    unsigned b_vol = (unsigned)seed | 1u;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        asm volatile("iadd3 %0, %0, %1, %0;" : "+r"(a) : "r"(b_vol));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = a; }

#elif OP == 2 && MODE == 1
    // IADD3 THROUGHPUT: 8 chains
    unsigned a0=tid+1,a1=tid+2,a2=tid+3,a3=tid+4,a4=tid+5,a5=tid+6,a6=tid+7,a7=tid+8;
    unsigned b_vol = (unsigned)seed | 1u;
    #pragma unroll
    for (int i = 0; i < N/8; i++) {
        asm volatile("iadd3 %0, %0, %1, %0;" : "+r"(a0) : "r"(b_vol));
        asm volatile("iadd3 %0, %0, %1, %0;" : "+r"(a1) : "r"(b_vol));
        asm volatile("iadd3 %0, %0, %1, %0;" : "+r"(a2) : "r"(b_vol));
        asm volatile("iadd3 %0, %0, %1, %0;" : "+r"(a3) : "r"(b_vol));
        asm volatile("iadd3 %0, %0, %1, %0;" : "+r"(a4) : "r"(b_vol));
        asm volatile("iadd3 %0, %0, %1, %0;" : "+r"(a5) : "r"(b_vol));
        asm volatile("iadd3 %0, %0, %1, %0;" : "+r"(a6) : "r"(b_vol));
        asm volatile("iadd3 %0, %0, %1, %0;" : "+r"(a7) : "r"(b_vol));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    unsigned s = a0^a1^a2^a3^a4^a5^a6^a7;
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = s; }

#elif OP == 3 && MODE == 0
    // DFMA LATENCY
    double a = (double)(tid + seed);
    double b = 1.0001;
    #pragma unroll
    for (int i = 0; i < N; i++) a = a * b + a;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = (unsigned)__double_as_longlong(a); }

#elif OP == 3 && MODE == 1
    // DFMA THROUGHPUT: 8 chains
    double a0=tid+1.,a1=tid+2.,a2=tid+3.,a3=tid+4.,a4=tid+5.,a5=tid+6.,a6=tid+7.,a7=tid+8.;
    double b = 1.0001;
    #pragma unroll
    for (int i = 0; i < N/8; i++) {
        a0 = a0*b + a0; a1 = a1*b + a1; a2 = a2*b + a2; a3 = a3*b + a3;
        a4 = a4*b + a4; a5 = a5*b + a5; a6 = a6*b + a6; a7 = a7*b + a7;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    double s = a0+a1+a2+a3+a4+a5+a6+a7;
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = (unsigned)__double_as_longlong(s); }

#elif OP == 4 && MODE == 0
    // LOP3.LUT LATENCY via inline asm (non-foldable)
    unsigned a = tid + seed;
    unsigned b_vol = (unsigned)seed;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        asm volatile("lop3.b32 %0, %0, %1, %0, 0x96;" : "+r"(a) : "r"(b_vol));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = a; }

#elif OP == 4 && MODE == 1
    // LOP3 THROUGHPUT
    unsigned a0=tid+1,a1=tid+2,a2=tid+3,a3=tid+4,a4=tid+5,a5=tid+6,a6=tid+7,a7=tid+8;
    unsigned b_vol = (unsigned)seed;
    #pragma unroll
    for (int i = 0; i < N/8; i++) {
        asm volatile("lop3.b32 %0, %0, %1, %0, 0x96;" : "+r"(a0) : "r"(b_vol));
        asm volatile("lop3.b32 %0, %0, %1, %0, 0x96;" : "+r"(a1) : "r"(b_vol));
        asm volatile("lop3.b32 %0, %0, %1, %0, 0x96;" : "+r"(a2) : "r"(b_vol));
        asm volatile("lop3.b32 %0, %0, %1, %0, 0x96;" : "+r"(a3) : "r"(b_vol));
        asm volatile("lop3.b32 %0, %0, %1, %0, 0x96;" : "+r"(a4) : "r"(b_vol));
        asm volatile("lop3.b32 %0, %0, %1, %0, 0x96;" : "+r"(a5) : "r"(b_vol));
        asm volatile("lop3.b32 %0, %0, %1, %0, 0x96;" : "+r"(a6) : "r"(b_vol));
        asm volatile("lop3.b32 %0, %0, %1, %0, 0x96;" : "+r"(a7) : "r"(b_vol));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    unsigned s = a0^a1^a2^a3^a4^a5^a6^a7;
    if (threadIdx.x == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = s; }
#endif
}
