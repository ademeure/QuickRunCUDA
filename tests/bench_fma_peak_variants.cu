// FMA variants peak: same methodology as bench_ffma_peak.
// 8 independent chains, INNER FMAs in inner unroll, OUTER outer with `#pragma unroll 1`.
//
// OP=0 : FFMA scalar f32
// OP=1 : FFMA2  (vec2 f32 packed)        — fma.rn.f32x2
// OP=2 : HFMA2  (vec2 f16 packed)
// OP=3 : HFMA  scalar f16 (16-bit)
// OP=4 : BFMA2 (vec2 bf16 packed)
// OP=5 : DFMA scalar f64

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#ifndef INNER
#define INNER 128
#endif
#ifndef OUTER
#define OUTER 100
#endif
#ifndef OP
#define OP 0
#endif

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

#if OP == 0  // FFMA scalar
    float v0=__int_as_float(tid+1)*1e-30f,v1=__int_as_float(tid+2)*1e-30f;
    float v2=__int_as_float(tid+3)*1e-30f,v3=__int_as_float(tid+4)*1e-30f;
    float v4=__int_as_float(tid+5)*1e-30f,v5=__int_as_float(tid+6)*1e-30f;
    float v6=__int_as_float(tid+7)*1e-30f,v7=__int_as_float(tid+8)*1e-30f;
    float y=1.5f;
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
        }
    }
    float sum = v0+v1+v2+v3+v4+v5+v6+v7;
    if (__float_as_int(sum) == seed) C[tid] = sum;
#endif

#if OP == 1  // FFMA2 vec2 f32 (manual inline asm to force packed)
    float2 v0={__int_as_float(tid+1)*1e-30f,__int_as_float(tid+2)*1e-30f};
    float2 v1={__int_as_float(tid+3)*1e-30f,__int_as_float(tid+4)*1e-30f};
    float2 v2={__int_as_float(tid+5)*1e-30f,__int_as_float(tid+6)*1e-30f};
    float2 v3={__int_as_float(tid+7)*1e-30f,__int_as_float(tid+8)*1e-30f};
    float2 v4={__int_as_float(tid+9)*1e-30f,__int_as_float(tid+10)*1e-30f};
    float2 v5={__int_as_float(tid+11)*1e-30f,__int_as_float(tid+12)*1e-30f};
    float2 v6={__int_as_float(tid+13)*1e-30f,__int_as_float(tid+14)*1e-30f};
    float2 v7={__int_as_float(tid+15)*1e-30f,__int_as_float(tid+16)*1e-30f};
    float2 y={1.5f, 1.5f};
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&v0) : "l"(*(unsigned long long*)&y), "l"(*(unsigned long long*)&v0));
            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&v1) : "l"(*(unsigned long long*)&y), "l"(*(unsigned long long*)&v1));
            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&v2) : "l"(*(unsigned long long*)&y), "l"(*(unsigned long long*)&v2));
            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&v3) : "l"(*(unsigned long long*)&y), "l"(*(unsigned long long*)&v3));
            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&v4) : "l"(*(unsigned long long*)&y), "l"(*(unsigned long long*)&v4));
            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&v5) : "l"(*(unsigned long long*)&y), "l"(*(unsigned long long*)&v5));
            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&v6) : "l"(*(unsigned long long*)&y), "l"(*(unsigned long long*)&v6));
            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&v7) : "l"(*(unsigned long long*)&y), "l"(*(unsigned long long*)&v7));
        }
    }
    float sum = v0.x+v0.y+v1.x+v1.y+v2.x+v2.y+v3.x+v3.y+v4.x+v4.y+v5.x+v5.y+v6.x+v6.y+v7.x+v7.y;
    if (__float_as_int(sum) == seed) C[tid] = sum;
#endif

#if OP == 2  // HFMA2 vec2 f16 — multiplier=1.5 to prevent FMA→ADD reduction
    __half2 v0=__float2half2_rn(1e-3f * (tid+1)), v1=__float2half2_rn(1e-3f * (tid+2));
    __half2 v2=__float2half2_rn(1e-3f * (tid+3)), v3=__float2half2_rn(1e-3f * (tid+4));
    __half2 v4=__float2half2_rn(1e-3f * (tid+5)), v5=__float2half2_rn(1e-3f * (tid+6));
    __half2 v6=__float2half2_rn(1e-3f * (tid+7)), v7=__float2half2_rn(1e-3f * (tid+8));
    __half2 y=__float2half2_rn(1.5f);
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            v0 = __hfma2(v0, y, v0); v1 = __hfma2(v1, y, v1);
            v2 = __hfma2(v2, y, v2); v3 = __hfma2(v3, y, v3);
            v4 = __hfma2(v4, y, v4); v5 = __hfma2(v5, y, v5);
            v6 = __hfma2(v6, y, v6); v7 = __hfma2(v7, y, v7);
        }
    }
    float sum = __low2float(v0)+__high2float(v0)+__low2float(v1)+__high2float(v1)
              +__low2float(v2)+__high2float(v2)+__low2float(v3)+__high2float(v3)
              +__low2float(v4)+__high2float(v4)+__low2float(v5)+__high2float(v5)
              +__low2float(v6)+__high2float(v6)+__low2float(v7)+__high2float(v7);
    if (__float_as_int(sum) == seed) C[tid] = sum;
#endif

#if OP == 3  // HFMA scalar f16 — half-the-rate of HFMA2
    __half v0=__float2half(1e-3f*(tid+1)),v1=__float2half(1e-3f*(tid+2));
    __half v2=__float2half(1e-3f*(tid+3)),v3=__float2half(1e-3f*(tid+4));
    __half v4=__float2half(1e-3f*(tid+5)),v5=__float2half(1e-3f*(tid+6));
    __half v6=__float2half(1e-3f*(tid+7)),v7=__float2half(1e-3f*(tid+8));
    __half y=__float2half(1.5f);
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            v0 = __hfma(v0, y, v0); v1 = __hfma(v1, y, v1);
            v2 = __hfma(v2, y, v2); v3 = __hfma(v3, y, v3);
            v4 = __hfma(v4, y, v4); v5 = __hfma(v5, y, v5);
            v6 = __hfma(v6, y, v6); v7 = __hfma(v7, y, v7);
        }
    }
    float sum = __half2float(v0)+__half2float(v1)+__half2float(v2)+__half2float(v3)
              +__half2float(v4)+__half2float(v5)+__half2float(v6)+__half2float(v7);
    if (__float_as_int(sum) == seed) C[tid] = sum;
#endif

#if OP == 4  // BFMA2 packed bf16x2 — multiplier=1.5
    __nv_bfloat162 v0=__float2bfloat162_rn(1e-3f*(tid+1)), v1=__float2bfloat162_rn(1e-3f*(tid+2));
    __nv_bfloat162 v2=__float2bfloat162_rn(1e-3f*(tid+3)), v3=__float2bfloat162_rn(1e-3f*(tid+4));
    __nv_bfloat162 v4=__float2bfloat162_rn(1e-3f*(tid+5)), v5=__float2bfloat162_rn(1e-3f*(tid+6));
    __nv_bfloat162 v6=__float2bfloat162_rn(1e-3f*(tid+7)), v7=__float2bfloat162_rn(1e-3f*(tid+8));
    __nv_bfloat162 y=__float2bfloat162_rn(1.5f);
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            v0 = __hfma2(v0, y, v0); v1 = __hfma2(v1, y, v1);
            v2 = __hfma2(v2, y, v2); v3 = __hfma2(v3, y, v3);
            v4 = __hfma2(v4, y, v4); v5 = __hfma2(v5, y, v5);
            v6 = __hfma2(v6, y, v6); v7 = __hfma2(v7, y, v7);
        }
    }
    float sum = __low2float(v0)+__high2float(v0)+__low2float(v1)+__high2float(v1)
              +__low2float(v2)+__high2float(v2)+__low2float(v3)+__high2float(v3)
              +__low2float(v4)+__high2float(v4)+__low2float(v5)+__high2float(v5)
              +__low2float(v6)+__high2float(v6)+__low2float(v7)+__high2float(v7);
    if (__float_as_int(sum) == seed) C[tid] = sum;
#endif

#if OP == 5  // DFMA scalar f64
    double v0=(tid+1)*1e-30,v1=(tid+2)*1e-30,v2=(tid+3)*1e-30,v3=(tid+4)*1e-30;
    double v4=(tid+5)*1e-30,v5=(tid+6)*1e-30,v6=(tid+7)*1e-30,v7=(tid+8)*1e-30;
    double y=1.5;
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
        }
    }
    double sum = v0+v1+v2+v3+v4+v5+v6+v7;
    if (__double_as_longlong(sum) == seed) C[tid] = (float)sum;
#endif
}
