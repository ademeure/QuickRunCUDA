// NVFP4 row reduction: int-domain (redux.sync.add) vs FP32 (SHFL chain)
// Single warp reduces a 256-element row = 16 blocks × 16 E2M1 values + 16 UE4M3 scales.
//
// Modes:
//   0: FP32 convert each E2M1×scale → fp32, accumulate per-thread, SHFL chain
//   1: Int per-block: decode E2M1→int4 (×2), sum 16 ints in registers, multiply by scale,
//      accumulate as fp32, SHFL chain
//   2: Full int: assume one common scale (max scale), decode E2M1×scale_ratio → int,
//      sum all in int, redux.sync.add, scale at end
//
// Each thread handles 8 elements (32 threads × 8 = 256).

#ifndef MODE
#define MODE 0
#endif
#ifndef N_ITERS
#define N_ITERS 1000
#endif

// Decode E2M1 (4-bit signed FP) to its ×2 integer representation.
// E2M1 4-bit: sign|exp|exp|mantissa
// Values: {0, 0.5, 1, 1.5, 2, 3, 4, 6} × sign
// Multiply by 2: {0, 1, 2, 3, 4, 6, 8, 12} × sign
__device__ __forceinline__ int decode_e2m1_x2(unsigned int code) {
    static const int lut[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    int mag = lut[code & 0x7];
    return (code & 0x8) ? -mag : mag;
}

// Decode UE4M3 (8-bit unsigned FP) to fp32.
__device__ __forceinline__ float decode_ue4m3(unsigned int code) {
    unsigned int exp = (code >> 3) & 0xF;
    unsigned int man = code & 0x7;
    if (exp == 0) {
        // subnormal: value = man * 2^-9
        return (float)man * (1.0f / 512.0f);
    } else {
        // normal: value = (1 + man/8) * 2^(exp-7)
        float val = 1.0f + (float)man * 0.125f;
        int shift = (int)exp - 7;
        if (shift >= 0) val *= (float)(1u << shift);
        else            val *= 1.0f / (float)(1u << (-shift));
        return val;
    }
}

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Pack 256 E2M1 values (1024 bits = 128 bytes = 32 dwords) and 16 UE4M3 scales
    // into the A buffer. Each thread reads its 8 nybbles + 0-1 scales.
    unsigned int* Au = (unsigned int*)A;
    unsigned char* As = (unsigned char*)(Au + 32);  // scales after data

    // Pre-fill A with deterministic data on iteration 0 only
    if (threadIdx.x == 0) {
        for (int i = 0; i < 32; i++) {
            // 8 nybbles per dword, repeating pattern
            unsigned int v = 0;
            for (int n = 0; n < 8; n++) {
                v |= ((i * 8 + n) & 0x7) << (n * 4);  // E2M1 codes 0..7 cycling
            }
            Au[i] = v;
        }
        for (int i = 0; i < 16; i++) {
            As[i] = 0x38 + (i & 0x7);  // UE4M3 ~1.0 area
        }
    }
    __syncwarp();

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Per-thread accumulator
    float facc = 0.0f;
    int   iacc = 0;
    int   imax_scale = 0;  // for mode 2 (find common scale)

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        // Each thread has 8 elements. Lane k handles elements [k*8 .. k*8+7].
        unsigned int dword = Au[threadIdx.x];  // 8 nybbles for this thread
        unsigned int my_block_idx = threadIdx.x / 2;  // 32 threads / 2 = 16 blocks
        unsigned int scale_byte = (threadIdx.x % 2 == 0) ? As[my_block_idx] : 0;
        // For correctness in mode 0, EVERY thread needs its block's scale; my mapping is
        // simplified — assume each pair of threads shares a scale block.
        if (threadIdx.x % 2 == 1) scale_byte = As[(threadIdx.x - 1) / 2];

#if MODE == 0
        // FP32 path: convert each element, accumulate
        float scale = decode_ue4m3(scale_byte);
        float partial = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            unsigned int code = (dword >> (n * 4)) & 0xF;
            int e2m1_x2 = decode_e2m1_x2(code);
            partial += (float)e2m1_x2 * 0.5f * scale;
        }
        facc += partial;
#elif MODE == 1
        // Hybrid: int sum per thread (no scale), then multiply by scale
        float scale = decode_ue4m3(scale_byte);
        int int_sum = 0;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            unsigned int code = (dword >> (n * 4)) & 0xF;
            int e2m1_x2 = decode_e2m1_x2(code);
            int_sum += e2m1_x2;
        }
        // int_sum is in units of 0.5/scale; multiply by 0.5*scale to get fp32
        facc += (float)int_sum * 0.5f * scale;
#elif MODE == 2
        // Full int path: assume common scale_factor=1 for testing speedup
        // (in practice would scale relative to max)
        int int_sum = 0;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            unsigned int code = (dword >> (n * 4)) & 0xF;
            int_sum += decode_e2m1_x2(code);
        }
        iacc += int_sum;
#endif
    }

    // Warp reduction
#if MODE == 0 || MODE == 1
    // SHFL chain on FP32
    float r = facc;
    r += __shfl_xor_sync(0xFFFFFFFF, r, 16);
    r += __shfl_xor_sync(0xFFFFFFFF, r,  8);
    r += __shfl_xor_sync(0xFFFFFFFF, r,  4);
    r += __shfl_xor_sync(0xFFFFFFFF, r,  2);
    r += __shfl_xor_sync(0xFFFFFFFF, r,  1);
#else
    // redux.sync.add on int
    int r;
    asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(iacc));
#endif

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[blockIdx.x * 2] = t1 - t0;
#if MODE == 0 || MODE == 1
        ((float*)C)[blockIdx.x * 2 + 1 + 1024] = r;
#else
        ((int*)C)[blockIdx.x * 2 + 1 + 1024] = r;
#endif
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N_ITERS=%d clk=%llu cy/iter=%.3f result=", MODE, N_ITERS, t1 - t0,
               (double)(t1-t0)/(double)N_ITERS);
#if MODE == 0 || MODE == 1
        printf("%.6f\n", r);
#else
        printf("%d (int)\n", r);
#endif
    }
}
