// Row-wise BF16 -> NVFP4 (e2m1) Quantization
//
// 1 block = 1 row. Each thread: 16 BF16 via 1x 256-bit load.
// Threads/row = ROW_DIM / 16.
//
// SEPARATE output pointers (no misalignment):
//   A = input  [num_rows, ROW_DIM] BF16 (contiguous, row-major)
//   B = output [num_rows, ROW_DIM/2] uint8 FP4 data (contiguous)
//   C = output [num_rows, ROW_DIM/16] uint8 e4m3 microscales
//   kernel_int_args[0] = num_rows
//   kernel_int_args[1] = scale_layout: 0=contiguous, 1=swizzled (cuBLAS e8 layout)
//
// Scale layouts:
//   Contiguous: scales[row][group] -- simple row-major
//   Swizzled:   cuBLAS-compatible interleaved layout for e8 scaling factors
//               groups within a 128B sector are interleaved across rows

#ifndef ROW_DIM
#define ROW_DIM 4096
#endif
#define THREADS_PER_ROW (ROW_DIM / 16)
#define WARPS_PER_ROW ((THREADS_PER_ROW + 31) / 32)
#define E2M1_MAX 6.0f
#define GROUPS_PER_ROW THREADS_PER_ROW

extern "C" __global__ void kernel(const float* __restrict__ A,
                                  float* __restrict__ B,
                                  float* __restrict__ C,
                                  int num_rows, int scale_layout, int unused_2) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ float smem[32];

    // =========================================================================
    // 1. 256-BIT LOAD: 16 BF16 = 32 bytes
    // =========================================================================
    const unsigned long long* src = (const unsigned long long*)A
        + (unsigned long long)row * (ROW_DIM / 4)
        + (unsigned long long)tid * 4;

    unsigned long long d0, d1, d2, d3;
    asm volatile(
        "ld.global.v4.u64 {%0,%1,%2,%3}, [%4];"
        : "=l"(d0), "=l"(d1), "=l"(d2), "=l"(d3)
        : "l"((unsigned long long)src)
    );

    // 8 bf16x2 pairs
    unsigned int bp0 = (unsigned int)d0,  bp1 = (unsigned int)(d0 >> 32);
    unsigned int bp2 = (unsigned int)d1,  bp3 = (unsigned int)(d1 >> 32);
    unsigned int bp4 = (unsigned int)d2,  bp5 = (unsigned int)(d2 >> 32);
    unsigned int bp6 = (unsigned int)d3,  bp7 = (unsigned int)(d3 >> 32);

    // =========================================================================
    // 2. BF16 -> F32 + absmax
    // =========================================================================
    float f0  = __int_as_float((bp0 & 0xFFFFu) << 16);
    float f1  = __int_as_float(bp0 & 0xFFFF0000u);
    float f2  = __int_as_float((bp1 & 0xFFFFu) << 16);
    float f3  = __int_as_float(bp1 & 0xFFFF0000u);
    float f4  = __int_as_float((bp2 & 0xFFFFu) << 16);
    float f5  = __int_as_float(bp2 & 0xFFFF0000u);
    float f6  = __int_as_float((bp3 & 0xFFFFu) << 16);
    float f7  = __int_as_float(bp3 & 0xFFFF0000u);
    float f8  = __int_as_float((bp4 & 0xFFFFu) << 16);
    float f9  = __int_as_float(bp4 & 0xFFFF0000u);
    float f10 = __int_as_float((bp5 & 0xFFFFu) << 16);
    float f11 = __int_as_float(bp5 & 0xFFFF0000u);
    float f12 = __int_as_float((bp6 & 0xFFFFu) << 16);
    float f13 = __int_as_float(bp6 & 0xFFFF0000u);
    float f14 = __int_as_float((bp7 & 0xFFFFu) << 16);
    float f15 = __int_as_float(bp7 & 0xFFFF0000u);

    float a0 = fmaxf(fabsf(f0),  fabsf(f1));
    float a1 = fmaxf(fabsf(f2),  fabsf(f3));
    float a2 = fmaxf(fabsf(f4),  fabsf(f5));
    float a3 = fmaxf(fabsf(f6),  fabsf(f7));
    float a4 = fmaxf(fabsf(f8),  fabsf(f9));
    float a5 = fmaxf(fabsf(f10), fabsf(f11));
    float a6 = fmaxf(fabsf(f12), fabsf(f13));
    float a7 = fmaxf(fabsf(f14), fabsf(f15));
    float tmax = fmaxf(fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)),
                        fmaxf(fmaxf(a4, a5), fmaxf(a6, a7)));

    // =========================================================================
    // 3. ROW ABSMAX
    // =========================================================================
    float wmax = tmax;
    wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFF, wmax, 16));
    wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFF, wmax, 8));
    wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFF, wmax, 4));
    wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFF, wmax, 2));
    wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFF, wmax, 1));

    if (lane == 0) smem[warp_id] = wmax;
    __syncthreads();

    float row_absmax;
    if (WARPS_PER_ROW <= 1) {
        row_absmax = smem[0];
    } else {
        float v = (tid < WARPS_PER_ROW) ? smem[tid] : 0.0f;
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 16));
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 8));
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 4));
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 2));
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 1));
        if (tid == 0) smem[0] = v;
        __syncthreads();
        row_absmax = smem[0];
    }

    // =========================================================================
    // 4. PRESCALE + CVT e2m1x2
    // =========================================================================
    float inv_scale;
    if (tmax > 0.0f) {
        asm volatile("rcp.approx.f32 %0, %1;" : "=f"(inv_scale) : "f"(tmax));
        inv_scale *= E2M1_MAX;
    } else {
        inv_scale = 0.0f;
    }

    f0  *= inv_scale; f1  *= inv_scale; f2  *= inv_scale; f3  *= inv_scale;
    f4  *= inv_scale; f5  *= inv_scale; f6  *= inv_scale; f7  *= inv_scale;
    f8  *= inv_scale; f9  *= inv_scale; f10 *= inv_scale; f11 *= inv_scale;
    f12 *= inv_scale; f13 *= inv_scale; f14 *= inv_scale; f15 *= inv_scale;

    unsigned short q0, q1, q2, q3, q4, q5, q6, q7;
    #define CVT_E2M1(dst, lo, hi) \
        asm volatile("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; mov.b16 %0, {t,0}; }" \
            : "=h"(dst) : "f"(lo), "f"(hi));
    CVT_E2M1(q0, f0, f1);   CVT_E2M1(q1, f2, f3);
    CVT_E2M1(q2, f4, f5);   CVT_E2M1(q3, f6, f7);
    CVT_E2M1(q4, f8, f9);   CVT_E2M1(q5, f10, f11);
    CVT_E2M1(q6, f12, f13); CVT_E2M1(q7, f14, f15);

    // =========================================================================
    // 5. STORE FP4 DATA -> B (separate pointer, naturally aligned)
    //    8 bytes per thread, row-major [num_rows, ROW_DIM/2]
    // =========================================================================
    unsigned char* fp4_out = (unsigned char*)B
        + (unsigned long long)row * (ROW_DIM / 2)
        + tid * 8;
    unsigned int pk0 = (q0&0xFF) | ((q1&0xFF)<<8) | ((q2&0xFF)<<16) | ((q3&0xFF)<<24);
    unsigned int pk1 = (q4&0xFF) | ((q5&0xFF)<<8) | ((q6&0xFF)<<16) | ((q7&0xFF)<<24);
    ((unsigned int*)fp4_out)[0] = pk0;
    ((unsigned int*)fp4_out)[1] = pk1;

    // =========================================================================
    // 6. STORE MICROSCALE -> C (separate pointer, naturally aligned)
    //    1 e4m3 byte per thread, layout selectable
    // =========================================================================
    float micro = (row_absmax > 0.0f) ? tmax / row_absmax : 0.0f;
    unsigned short me;
    asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}" : "=h"(me) : "f"(micro), "f"(0.0f));
    unsigned char scale_byte = (unsigned char)(me & 0xFF);

    unsigned char* scale_out = (unsigned char*)C;
    if (scale_layout == 0) {
        // Contiguous: scales[row][group] -- simple row-major
        scale_out[row * GROUPS_PER_ROW + tid] = scale_byte;
    } else {
        // Swizzled: cuBLAS e8 scaling factor layout
        // Groups within a 128-byte sector are interleaved across rows
        // Layout: for each 128-scale chunk, scales are stored as:
        //   [row0_g0, row1_g0, row2_g0, ..., row127_g0, row0_g1, row1_g1, ...]
        // This matches how cuBLAS reads scales for dequant during GEMM
        const int scales_per_sector = 128;
        int group_in_row = tid;
        int sector = group_in_row / scales_per_sector;
        int pos_in_sector = group_in_row % scales_per_sector;
        // Interleave rows within each sector block of num_rows * scales_per_sector
        int block_offset = sector * (num_rows * scales_per_sector);
        int offset = block_offset + pos_in_sector * num_rows + row;
        scale_out[offset] = scale_byte;
    }

    // Row absmax: store to a separate location (first float of each row in scale_out)
    // Actually -- use the end of the scale array as row_absmax storage
    // Or better: let the caller provide a 4th pointer. For now, use smem[0] already set.
    // Store row absmax at scale_out[num_rows * GROUPS_PER_ROW + row * 4 .. +3]
    if (tid == 0) {
        float* row_scale_out = (float*)(scale_out + (unsigned long long)num_rows * GROUPS_PER_ROW);
        row_scale_out[row] = row_absmax;
    }
}

extern "C" __global__ void init(float* A, float* B, float* C,
                                int num_rows, int unused_1, int unused_2) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int* a = (unsigned int*)A;
    int total = num_rows * (ROW_DIM / 2);
    if (gid < total) {
        unsigned int s = gid * 2654435761u + 1;
        s = s * 1664525u + 1013904223u;
        a[gid] = ((s >> 8) & 0xFFFF0000u) | (s & 0xFFFF);
    }
}
