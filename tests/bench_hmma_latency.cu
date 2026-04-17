// HMMA latency: single-chain mma.sync with accumulator dep.
// Contrast with throughput test (8+ chains at 577 TFLOPS).

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef N
#define N 1024
#endif
#ifndef MODE
#define MODE 0  // 0=latency (1 chain), 1=throughput (8 chains)
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = threadIdx.x;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    unsigned a0 = 0x3c003c00 ^ tid;
    unsigned a1 = 0x3c003c00 ^ (tid<<1);
    unsigned a2 = 0x3c003c00 ^ (tid<<2);
    unsigned a3 = 0x3c003c00 ^ (tid<<3);
    unsigned b0 = 0x3c003c00 ^ (tid<<4);
    unsigned b1 = 0x3c003c00 ^ (tid<<5);

#if MODE == 0
    // LATENCY: single chain, accumulator feeds back
    float c0=tid+1.f, c1=tid+2.f, c2=tid+3.f, c3=tid+4.f;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    float s = c0+c1+c2+c3;
    if (tid == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = __float_as_int(s); }

#elif MODE == 1
    // THROUGHPUT: 8 chains
    float c[8][4];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        c[k][0] = tid+k*4+1.f; c[k][1] = tid+k*4+2.f;
        c[k][2] = tid+k*4+3.f; c[k][3] = tid+k*4+4.f;
    }
    #pragma unroll
    for (int i = 0; i < N/8; i++) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    float s = 0;
    for (int k = 0; k < 8; k++) s += c[k][0]+c[k][1]+c[k][2]+c[k][3];
    if (tid == 0) { ((unsigned long long*)C)[0] = t1-t0; C[2] = __float_as_int(s); }
#endif
}
