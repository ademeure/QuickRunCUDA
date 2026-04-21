// V6 H4: INT cvt latency on B300
// MODE 0: cvt FP32 → INT8 (sat)
// MODE 1: cvt FP32 → INT4 (sat) — bit packing
// MODE 2: cvt INT8 → FP32 (dequant)
// MODE 3: cvt INT4 → FP32 (dequant)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float base_a = (float)(threadIdx.x + 1) * 0.5f;
    int base_i = threadIdx.x + 1;
    int v0=0, v1=0, v2=0, v3=0, v4=0, v5=0, v6=0, v7=0;
    float f0=0, f1=0, f2=0, f3=0, f4=0, f5=0, f6=0, f7=0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        float a = base_a + (float)i * 0.001f;
        int x = base_i + i;
#if MODE == 0
        // FP32 → INT8 (sat)
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(v0) : "f"(a));
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(v1) : "f"(a + 0.1f));
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(v2) : "f"(a + 0.2f));
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(v3) : "f"(a + 0.3f));
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(v4) : "f"(a + 0.4f));
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(v5) : "f"(a + 0.5f));
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(v6) : "f"(a + 0.6f));
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(v7) : "f"(a + 0.7f));
#elif MODE == 1
        // FP32 → INT4 (sat)
        asm volatile("cvt.rni.sat.s4.f32 %0, %1;" : "=r"(v0) : "f"(a));
        asm volatile("cvt.rni.sat.s4.f32 %0, %1;" : "=r"(v1) : "f"(a + 0.1f));
        asm volatile("cvt.rni.sat.s4.f32 %0, %1;" : "=r"(v2) : "f"(a + 0.2f));
        asm volatile("cvt.rni.sat.s4.f32 %0, %1;" : "=r"(v3) : "f"(a + 0.3f));
        asm volatile("cvt.rni.sat.s4.f32 %0, %1;" : "=r"(v4) : "f"(a + 0.4f));
        asm volatile("cvt.rni.sat.s4.f32 %0, %1;" : "=r"(v5) : "f"(a + 0.5f));
        asm volatile("cvt.rni.sat.s4.f32 %0, %1;" : "=r"(v6) : "f"(a + 0.6f));
        asm volatile("cvt.rni.sat.s4.f32 %0, %1;" : "=r"(v7) : "f"(a + 0.7f));
#elif MODE == 2
        // INT8 → FP32 (dequant)
        asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(f0) : "r"(x));
        asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(f1) : "r"(x + 1));
        asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(f2) : "r"(x + 2));
        asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(f3) : "r"(x + 3));
        asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(f4) : "r"(x + 4));
        asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(f5) : "r"(x + 5));
        asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(f6) : "r"(x + 6));
        asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(f7) : "r"(x + 7));
#elif MODE == 3
        // INT4 → FP32 (dequant)
        asm volatile("cvt.rn.f32.s4 %0, %1;" : "=f"(f0) : "r"(x));
        asm volatile("cvt.rn.f32.s4 %0, %1;" : "=f"(f1) : "r"(x + 1));
        asm volatile("cvt.rn.f32.s4 %0, %1;" : "=f"(f2) : "r"(x + 2));
        asm volatile("cvt.rn.f32.s4 %0, %1;" : "=f"(f3) : "r"(x + 3));
        asm volatile("cvt.rn.f32.s4 %0, %1;" : "=f"(f4) : "r"(x + 4));
        asm volatile("cvt.rn.f32.s4 %0, %1;" : "=f"(f5) : "r"(x + 5));
        asm volatile("cvt.rn.f32.s4 %0, %1;" : "=f"(f6) : "r"(x + 6));
        asm volatile("cvt.rn.f32.s4 %0, %1;" : "=f"(f7) : "r"(x + 7));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    int isum = v0+v1+v2+v3+v4+v5+v6+v7;
    float fsum = f0+f1+f2+f3+f4+f5+f6+f7;
    if (isum + (int)fsum == 0xCAFEBABE) C[blockIdx.x] = (float)isum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d cy/iter=%.3f cy/cvt=%.3f\n",
               MODE, ITERS, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/8.0);
    }
}
