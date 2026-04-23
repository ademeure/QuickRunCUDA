// True register pressure test: declare N FMA-chain accumulators, all kept live
// across the loop. Use macros/template-trick to vary N via preprocessor.
//
// -H "#define N_REGS <n>"  -- N independent FMA chains (each needs ~1 register)
// -H "#define ITERS <n>"    -- chain depth

#ifndef N_REGS
#define N_REGS 8
#endif
#ifndef ITERS
#define ITERS 1024
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif

// Generate N_REGS unique vars and FMA them
#define DECLARE_N(prefix, val) \
    float prefix##_0=val+0.001f, prefix##_1=val+0.002f, prefix##_2=val+0.003f, prefix##_3=val+0.004f, \
          prefix##_4=val+0.005f, prefix##_5=val+0.006f, prefix##_6=val+0.007f, prefix##_7=val+0.008f, \
          prefix##_8=val+0.009f, prefix##_9=val+0.010f, prefix##_10=val+0.011f, prefix##_11=val+0.012f, \
          prefix##_12=val+0.013f, prefix##_13=val+0.014f, prefix##_14=val+0.015f, prefix##_15=val+0.016f, \
          prefix##_16=val+0.017f, prefix##_17=val+0.018f, prefix##_18=val+0.019f, prefix##_19=val+0.020f, \
          prefix##_20=val+0.021f, prefix##_21=val+0.022f, prefix##_22=val+0.023f, prefix##_23=val+0.024f, \
          prefix##_24=val+0.025f, prefix##_25=val+0.026f, prefix##_26=val+0.027f, prefix##_27=val+0.028f, \
          prefix##_28=val+0.029f, prefix##_29=val+0.030f, prefix##_30=val+0.031f, prefix##_31=val+0.032f, \
          prefix##_32=val+0.033f, prefix##_33=val+0.034f, prefix##_34=val+0.035f, prefix##_35=val+0.036f, \
          prefix##_36=val+0.037f, prefix##_37=val+0.038f, prefix##_38=val+0.039f, prefix##_39=val+0.040f, \
          prefix##_40=val+0.041f, prefix##_41=val+0.042f, prefix##_42=val+0.043f, prefix##_43=val+0.044f, \
          prefix##_44=val+0.045f, prefix##_45=val+0.046f, prefix##_46=val+0.047f, prefix##_47=val+0.048f, \
          prefix##_48=val+0.049f, prefix##_49=val+0.050f, prefix##_50=val+0.051f, prefix##_51=val+0.052f, \
          prefix##_52=val+0.053f, prefix##_53=val+0.054f, prefix##_54=val+0.055f, prefix##_55=val+0.056f, \
          prefix##_56=val+0.057f, prefix##_57=val+0.058f, prefix##_58=val+0.059f, prefix##_59=val+0.060f, \
          prefix##_60=val+0.061f, prefix##_61=val+0.062f, prefix##_62=val+0.063f, prefix##_63=val+0.064f

#define FMA_N(prefix, k, b) \
    prefix##_0 = prefix##_0 * k + b; prefix##_1 = prefix##_1 * k + b; \
    prefix##_2 = prefix##_2 * k + b; prefix##_3 = prefix##_3 * k + b; \
    prefix##_4 = prefix##_4 * k + b; prefix##_5 = prefix##_5 * k + b; \
    prefix##_6 = prefix##_6 * k + b; prefix##_7 = prefix##_7 * k + b
#define FMA_N16(prefix, k, b) \
    FMA_N(prefix, k, b); \
    prefix##_8 = prefix##_8 * k + b; prefix##_9 = prefix##_9 * k + b; \
    prefix##_10 = prefix##_10 * k + b; prefix##_11 = prefix##_11 * k + b; \
    prefix##_12 = prefix##_12 * k + b; prefix##_13 = prefix##_13 * k + b; \
    prefix##_14 = prefix##_14 * k + b; prefix##_15 = prefix##_15 * k + b
#define FMA_N32(prefix, k, b) \
    FMA_N16(prefix, k, b); \
    prefix##_16 = prefix##_16 * k + b; prefix##_17 = prefix##_17 * k + b; \
    prefix##_18 = prefix##_18 * k + b; prefix##_19 = prefix##_19 * k + b; \
    prefix##_20 = prefix##_20 * k + b; prefix##_21 = prefix##_21 * k + b; \
    prefix##_22 = prefix##_22 * k + b; prefix##_23 = prefix##_23 * k + b; \
    prefix##_24 = prefix##_24 * k + b; prefix##_25 = prefix##_25 * k + b; \
    prefix##_26 = prefix##_26 * k + b; prefix##_27 = prefix##_27 * k + b; \
    prefix##_28 = prefix##_28 * k + b; prefix##_29 = prefix##_29 * k + b; \
    prefix##_30 = prefix##_30 * k + b; prefix##_31 = prefix##_31 * k + b
#define FMA_N64(prefix, k, b) \
    FMA_N32(prefix, k, b); \
    prefix##_32 = prefix##_32 * k + b; prefix##_33 = prefix##_33 * k + b; \
    prefix##_34 = prefix##_34 * k + b; prefix##_35 = prefix##_35 * k + b; \
    prefix##_36 = prefix##_36 * k + b; prefix##_37 = prefix##_37 * k + b; \
    prefix##_38 = prefix##_38 * k + b; prefix##_39 = prefix##_39 * k + b; \
    prefix##_40 = prefix##_40 * k + b; prefix##_41 = prefix##_41 * k + b; \
    prefix##_42 = prefix##_42 * k + b; prefix##_43 = prefix##_43 * k + b; \
    prefix##_44 = prefix##_44 * k + b; prefix##_45 = prefix##_45 * k + b; \
    prefix##_46 = prefix##_46 * k + b; prefix##_47 = prefix##_47 * k + b; \
    prefix##_48 = prefix##_48 * k + b; prefix##_49 = prefix##_49 * k + b; \
    prefix##_50 = prefix##_50 * k + b; prefix##_51 = prefix##_51 * k + b; \
    prefix##_52 = prefix##_52 * k + b; prefix##_53 = prefix##_53 * k + b; \
    prefix##_54 = prefix##_54 * k + b; prefix##_55 = prefix##_55 * k + b; \
    prefix##_56 = prefix##_56 * k + b; prefix##_57 = prefix##_57 * k + b; \
    prefix##_58 = prefix##_58 * k + b; prefix##_59 = prefix##_59 * k + b; \
    prefix##_60 = prefix##_60 * k + b; prefix##_61 = prefix##_61 * k + b; \
    prefix##_62 = prefix##_62 * k + b; prefix##_63 = prefix##_63 * k + b

#define SUM_N(prefix) (prefix##_0 + prefix##_1 + prefix##_2 + prefix##_3 + \
    prefix##_4 + prefix##_5 + prefix##_6 + prefix##_7)
#define SUM_N16(prefix) (SUM_N(prefix) + prefix##_8 + prefix##_9 + prefix##_10 + \
    prefix##_11 + prefix##_12 + prefix##_13 + prefix##_14 + prefix##_15)
#define SUM_N32(prefix) (SUM_N16(prefix) + prefix##_16 + prefix##_17 + prefix##_18 + \
    prefix##_19 + prefix##_20 + prefix##_21 + prefix##_22 + prefix##_23 + \
    prefix##_24 + prefix##_25 + prefix##_26 + prefix##_27 + prefix##_28 + \
    prefix##_29 + prefix##_30 + prefix##_31)
#define SUM_N64(prefix) (SUM_N32(prefix) + prefix##_32 + prefix##_33 + prefix##_34 + \
    prefix##_35 + prefix##_36 + prefix##_37 + prefix##_38 + prefix##_39 + \
    prefix##_40 + prefix##_41 + prefix##_42 + prefix##_43 + prefix##_44 + \
    prefix##_45 + prefix##_46 + prefix##_47 + prefix##_48 + prefix##_49 + \
    prefix##_50 + prefix##_51 + prefix##_52 + prefix##_53 + prefix##_54 + \
    prefix##_55 + prefix##_56 + prefix##_57 + prefix##_58 + prefix##_59 + \
    prefix##_60 + prefix##_61 + prefix##_62 + prefix##_63)

extern "C" __global__ __launch_bounds__(32, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (blockIdx.x != 0) return;
    int lane = threadIdx.x;
    float val = (float)(lane + seed) * 0.001f + 1.0f;
    float k = (float)(seed + 1) * 0.0001f + 1.00001f;
    float b = (float)(u1 + 1) * 0.0001f + 1.0f;

    DECLARE_N(v, val);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if N_REGS == 8
        FMA_N(v, k, b);
#elif N_REGS == 16
        FMA_N16(v, k, b);
#elif N_REGS == 32
        FMA_N32(v, k, b);
#elif N_REGS == 64
        FMA_N64(v, k, b);
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

#if N_REGS == 8
    float sum = SUM_N(v);
#elif N_REGS == 16
    float sum = SUM_N16(v);
#elif N_REGS == 32
    float sum = SUM_N32(v);
#elif N_REGS == 64
    float sum = SUM_N64(v);
#endif

    if (sum == 0xDEADBEEF) C[lane] = sum;
    if (lane == 0) ((unsigned long long*)C)[1024] = (t1 - t0);
}
