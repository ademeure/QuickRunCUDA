// mma.sync BF16 ILP: 1, 2, 4 independent acc tiles per warp
#ifndef ILP
#define ILP 1
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int a0 = (threadIdx.x ^ u2) | 0x11111111u;
    unsigned int a1 = (threadIdx.x ^ u2) | 0x22222222u;
    unsigned int a2 = (threadIdx.x ^ u2) | 0x33333333u;
    unsigned int a3 = (threadIdx.x ^ u2) | 0x44444444u;
    unsigned int b0 = (threadIdx.x ^ u2) | 0x55555555u;
    unsigned int b1 = (threadIdx.x ^ u2) | 0x66666666u;
    float c0=0,c1=0,c2=0,c3=0;
#if ILP >= 2
    float d0=0,d1=0,d2=0,d3=0;
#endif
#if ILP >= 4
    float e0=0,e1=0,e2=0,e3=0;
    float f0=0,f1=0,f2=0,f3=0;
#endif
#if ILP >= 8
    float g0=0,g1=0,g2=0,g3=0;
    float h0=0,h1=0,h2=0,h3=0;
    float i0=0,i1=0,i2=0,i3=0;
    float j0=0,j1=0,j2=0,j3=0;
#endif

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        a0 ^= i;
#define MMA(a,b,c,d,e,f) \
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
            : "+f"(a),"+f"(b),"+f"(c),"+f"(d) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        MMA(c0,c1,c2,c3,_,_);
#if ILP >= 2
        MMA(d0,d1,d2,d3,_,_);
#endif
#if ILP >= 4
        MMA(e0,e1,e2,e3,_,_);
        MMA(f0,f1,f2,f3,_,_);
#endif
#if ILP >= 8
        MMA(g0,g1,g2,g3,_,_);
        MMA(h0,h1,h2,h3,_,_);
        MMA(i0,i1,i2,i3,_,_);
        MMA(j0,j1,j2,j3,_,_);
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sink = c0+c1+c2+c3;
#if ILP >= 2
    sink += d0+d1+d2+d3;
#endif
#if ILP >= 4
    sink += e0+e1+e2+e3+f0+f1+f2+f3;
#endif
#if ILP >= 8
    sink += g0+g1+g2+g3+h0+h1+h2+h3+i0+i1+i2+i3+j0+j1+j2+j3;
#endif
    if ((int)sink == seed) C[blockIdx.x] = sink;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("ILP=%d clk=%llu cy/iter=%.2f cy_per_mma=%.3f\n",
               ILP, t1-t0, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/(double)ILP);
    }
}
