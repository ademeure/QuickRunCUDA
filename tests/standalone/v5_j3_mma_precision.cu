// V5 J3: BF16 mma.sync m16n8k16 numerical precision vs FP32 reference
// Generate random A (16x16 BF16) and B (16x8 BF16); compute C via:
//   1. m16n8k16 BF16 mma.sync (FP32 accumulator)
//   2. Reference: BF16 → FP32, FP32 GEMM
// Measure max ULP error

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void mma_kernel(__nv_bfloat16* A, __nv_bfloat16* B, float* C_mma, float* C_ref) {
    if (threadIdx.x >= 32) return;
    int lane = threadIdx.x;

    // Load A: 16x16 BF16, 8 elem per thread (2 fragments × 4 elem)
    // Layout per PTX m16n8k16 BF16:
    //   A: row=lane%16, col=lane/16 + 0,8 → 4 elem per thread? Actually 8 elem = 4×2
    // For simplicity: each thread loads 8 BF16 = 4 uint32 (2 BF16 each)
    unsigned int a0 = ((unsigned int*)A)[lane * 4 + 0];
    unsigned int a1 = ((unsigned int*)A)[lane * 4 + 1];
    unsigned int a2 = ((unsigned int*)A)[lane * 4 + 2];
    unsigned int a3 = ((unsigned int*)A)[lane * 4 + 3];
    // B: 16x8 BF16, 4 elem per thread = 2 uint32
    unsigned int b0 = ((unsigned int*)B)[lane * 2 + 0];
    unsigned int b1 = ((unsigned int*)B)[lane * 2 + 1];

    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));

    // Layout per docs: each thread holds 4 elem at:
    //   row = lane / 4, col = (lane % 4) * 2 (and +1)
    // Write contiguously by thread
    C_mma[lane * 4 + 0] = c0;
    C_mma[lane * 4 + 1] = c1;
    C_mma[lane * 4 + 2] = c2;
    C_mma[lane * 4 + 3] = c3;
}

__global__ void ref_kernel(__nv_bfloat16* A, __nv_bfloat16* B, float* C) {
    // Reference: 16x8 = 128 outputs; each thread computes 4 (matching mma layout)
    int lane = threadIdx.x;
    if (lane >= 32) return;

    // Compute c[lane*4+0..3] using same layout as mma
    int row_base = lane / 4;       // 0..7
    int col_base = (lane % 4) * 2; // 0,2,4,6
    // Outputs at (row_base, col_base+0), (row_base, col_base+1),
    //            (row_base+8, col_base+0), (row_base+8, col_base+1)
    int rows[4] = {row_base, row_base, row_base + 8, row_base + 8};
    int cols[4] = {col_base, col_base + 1, col_base, col_base + 1};

    for (int o = 0; o < 4; o++) {
        float sum = 0;
        for (int k = 0; k < 16; k++) {
            float a = __bfloat162float(A[rows[o] * 16 + k]);
            float b = __bfloat162float(B[k * 8 + cols[o]]);
            sum += a * b;
        }
        C[lane * 4 + o] = sum;
    }
}

int main() {
    cudaSetDevice(0);
    __nv_bfloat16 *A, *B;
    float *C_mma, *C_ref;
    cudaMallocManaged(&A, 16 * 16 * sizeof(__nv_bfloat16));
    cudaMallocManaged(&B, 16 * 8 * sizeof(__nv_bfloat16));
    cudaMallocManaged(&C_mma, 16 * 8 * sizeof(float));
    cudaMallocManaged(&C_ref, 16 * 8 * sizeof(float));

    // Generate random BF16 inputs in [-1, 1]
    srand(42);
    for (int i = 0; i < 256; i++) A[i] = __float2bfloat16(2.0f * rand() / RAND_MAX - 1.0f);
    for (int i = 0; i < 128; i++) B[i] = __float2bfloat16(2.0f * rand() / RAND_MAX - 1.0f);

    mma_kernel<<<1, 32>>>(A, B, C_mma, nullptr);
    ref_kernel<<<1, 32>>>(A, B, C_ref);
    cudaDeviceSynchronize();

    // Compare
    float max_abs_err = 0;
    float max_rel_err = 0;
    int max_ulp = 0;
    for (int i = 0; i < 128; i++) {
        float diff = fabsf(C_mma[i] - C_ref[i]);
        if (diff > max_abs_err) max_abs_err = diff;
        if (fabsf(C_ref[i]) > 0.001f) {
            float rel = diff / fabsf(C_ref[i]);
            if (rel > max_rel_err) max_rel_err = rel;
        }
        // ULP = abs(int32(mma) - int32(ref))
        unsigned int mma_bits = *(unsigned int*)&C_mma[i];
        unsigned int ref_bits = *(unsigned int*)&C_ref[i];
        int ulp = abs((int)mma_bits - (int)ref_bits);
        if (ulp > max_ulp) max_ulp = ulp;
    }

    printf("BF16 m16n8k16 MMA vs FP32 reference:\n");
    printf("  max abs error:  %.6e\n", max_abs_err);
    printf("  max rel error:  %.6e\n", max_rel_err);
    printf("  max ULP error:  %d\n", max_ulp);
    printf("  Sample C[0]: mma=%.6f ref=%.6f\n", C_mma[0], C_ref[0]);

    return 0;
}
