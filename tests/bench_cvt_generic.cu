// Generic CVT throughput benchmark
// Define CVT_BODY and CVT_OPS_PER_ITER via -H header

#ifndef UNROLL
#define UNROLL 8
#endif

#ifndef CVT_BODY
#define CVT_BODY \
    asm volatile( \
        "cvt.rn.f16.f32 %0, %4;\n\t" \
        "cvt.rn.f16.f32 %1, %5;\n\t" \
        "cvt.rn.f16.f32 %2, %6;\n\t" \
        "cvt.rn.f16.f32 %3, %7;\n\t" \
        : "=h"(h0), "=h"(h1), "=h"(h2), "=h"(h3) \
        : "f"(f0), "f"(f1), "f"(f2), "f"(f3) \
    );
#endif

#ifndef CVT_OPS_PER_ITER
#define CVT_OPS_PER_ITER 4
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int unused_1, int unused_2) {
    unsigned int tid = threadIdx.x;
    float tid_f = (float)(tid & 0xFF) * 0.001f;
    unsigned short tid_h = (unsigned short)(tid & 0xFF);

    unsigned short h0=0, h1=0, h2=0, h3=0, h4=0, h5=0, h6=0, h7=0;
    unsigned int r0=0, r1=0, r2=0, r3=0, r4=0, r5=0, r6=0, r7=0;
    unsigned int acc = 0;
    unsigned int rbits = tid * 2654435761u + 1;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            float fi = __int_as_float((i + j) ^ tid);
            float f0=1.0f+fi+tid_f, f1=2.0f+fi, f2=0.5f+fi+tid_f, f3=1.5f+fi;
            float f4=3.0f+fi, f5=0.25f+fi+tid_f, f6=4.0f+fi, f7=0.125f+fi+tid_f;
            unsigned short ih0 = (unsigned short)((i+j) ^ 0x3838) ^ tid_h;
            unsigned short ih1 = (unsigned short)((i+j) ^ 0x3839) ^ tid_h;
            unsigned short ih2 = (unsigned short)((i+j) ^ 0x3938) ^ tid_h;
            unsigned short ih3 = (unsigned short)((i+j) ^ 0x3939) ^ tid_h;
            unsigned int ir0 = (unsigned int)(i+j) ^ 0x3C003C00u ^ tid;
            unsigned int ir1 = (unsigned int)(i+j) ^ 0x3C013C01u ^ tid;
            unsigned int ir2 = (unsigned int)(i+j) ^ 0x3C023C02u ^ tid;
            unsigned int ir3 = (unsigned int)(i+j) ^ 0x3C033C03u ^ tid;
            rbits = rbits * 1664525u + 1013904223u;

            CVT_BODY

            acc ^= (unsigned int)(h0^h1^h2^h3^h4^h5^h6^h7) ^ r0^r1^r2^r3^r4^r5^r6^r7;
        }
    }

    if (tid >= (unsigned int)blockDim.x) { ((unsigned int*)C)[tid] = acc; }
}
