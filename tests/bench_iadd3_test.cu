#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef N
#define N 1024
#endif
#ifndef MODE
#define MODE 0
#endif
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = threadIdx.x;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    unsigned b = (unsigned)seed | 1u;
#if MODE == 0
    unsigned a = tid + seed;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        unsigned t = a + b + (a ^ b);  // non-closed-form; should emit IADD3
        a = t;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (tid == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = a; }
#elif MODE == 1
    unsigned a0=tid+1,a1=tid+2,a2=tid+3,a3=tid+4,a4=tid+5,a5=tid+6,a6=tid+7,a7=tid+8;
    #pragma unroll
    for (int i = 0; i < N/8; i++) {
        a0 = a0 + b + (a0 ^ b);
        a1 = a1 + b + (a1 ^ b);
        a2 = a2 + b + (a2 ^ b);
        a3 = a3 + b + (a3 ^ b);
        a4 = a4 + b + (a4 ^ b);
        a5 = a5 + b + (a5 ^ b);
        a6 = a6 + b + (a6 ^ b);
        a7 = a7 + b + (a7 ^ b);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    unsigned s = a0^a1^a2^a3^a4^a5^a6^a7;
    if (tid == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = s; }
#endif
}
