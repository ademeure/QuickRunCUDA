// Integer/FP division — is it a single instruction or emulated?

#ifndef UNROLL
#define UNROLL 8
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(256, 2)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int a = threadIdx.x + 1;
    unsigned int b = 7;
    int ia = (int)a, ib = 3;
    float fa = (float)(threadIdx.x + 1);
    float fb = 1.7f;
    double da = 1.7, db = 1.5;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // u32 divide
            a = a / b;
            b = b + 1;
#elif OP == 1  // s32 divide
            ia = ia / ib;
            ib = ib + 1;
#elif OP == 2  // u32 modulo
            a = a % b;
            b = b + 2;
#elif OP == 3  // f32 divide (rn default)
            fa = fa / fb;
            fb = fb + 0.0001f;
#elif OP == 4  // f32 precise divide (div.rn.f32)
            asm volatile("div.rn.f32 %0, %0, %1;" : "+f"(fa) : "f"(fb));
#elif OP == 5  // f32 approx divide (div.approx.f32)
            asm volatile("div.approx.f32 %0, %0, %1;" : "+f"(fa) : "f"(fb));
#elif OP == 6  // f64 divide
            da = da / db;
            db = db + 0.01;
#elif OP == 7  // f32 sqrt precise
            asm volatile("sqrt.rn.f32 %0, %0;" : "+f"(fa));
            fa += 0.01f;
#elif OP == 8  // f32 rem
            asm volatile("rem.f32 %0, %0, %1;" : "+f"(fa) : "f"(fb));
#endif
        }
    }
    if (a == (unsigned)seed || __float_as_int(fa) == seed)
        C[blockIdx.x * blockDim.x + threadIdx.x] = fa + __int_as_float(a) + (float)da;
}
