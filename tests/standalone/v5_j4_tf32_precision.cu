// V5 J4: TF32 vs FP32 multiply precision
// Compare mantissa precision: TF32 has 10-bit, FP32 has 23-bit
// Test: a * b where exact product needs >10 bits of mantissa precision
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void fp32_mul(float* a, float* b, float* out, int n) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < n) out[gtid] = a[gtid] * b[gtid];
}

__global__ void tf32_mul(float* a, float* b, float* out, int n) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < n) {
        // Convert FP32 → TF32 (round 23-bit mantissa to 10-bit) then multiply
        float a_tf32, b_tf32;
        asm("cvt.rna.tf32.f32 %0, %1;" : "=f"(a_tf32) : "f"(a[gtid]));
        asm("cvt.rna.tf32.f32 %0, %1;" : "=f"(b_tf32) : "f"(b[gtid]));
        // Multiply in fp32 (TF32 only affects MMA accumulation; cvt for storage)
        out[gtid] = a_tf32 * b_tf32;
    }
}

int main() {
    cudaSetDevice(0);
    int n = 1024;
    float *a, *b, *out_fp32, *out_tf32;
    cudaMallocManaged(&a, n * sizeof(float));
    cudaMallocManaged(&b, n * sizeof(float));
    cudaMallocManaged(&out_fp32, n * sizeof(float));
    cudaMallocManaged(&out_tf32, n * sizeof(float));

    // Generate values within TF32's resolution (10-bit mantissa)
    // Variations at 1e-3 to 1e-4 range will be resolved by TF32 but with quantization
    srand(42);
    for (int i = 0; i < n; i++) {
        a[i] = 1.0f + (float)i * 1e-4f;  // 1.0001, 1.0002, ... 1.1023
        b[i] = 1.0f - (float)i * 1e-4f;
    }

    fp32_mul<<<4, 256>>>(a, b, out_fp32, n);
    tf32_mul<<<4, 256>>>(a, b, out_tf32, n);
    cudaDeviceSynchronize();

    // Compare
    double max_abs_err = 0;
    double max_rel_err = 0;
    int max_ulp = 0;
    for (int i = 0; i < n; i++) {
        // FP64 reference
        double ref = (double)a[i] * (double)b[i];
        double fp32_err = fabs((double)out_fp32[i] - ref);
        double tf32_err = fabs((double)out_tf32[i] - ref);
        if (tf32_err > max_abs_err) max_abs_err = tf32_err;
        if (ref > 0.001) {
            double rel = tf32_err / fabs(ref);
            if (rel > max_rel_err) max_rel_err = rel;
        }
        unsigned int fp_bits = *(unsigned int*)&out_fp32[i];
        unsigned int tf_bits = *(unsigned int*)&out_tf32[i];
        int ulp = abs((int)fp_bits - (int)tf_bits);
        if (ulp > max_ulp) max_ulp = ulp;
    }
    printf("TF32 vs FP32 multiply precision (1024 random near-1.0 values):\n");
    printf("  TF32 mantissa: 10 bits (FP32 has 23 bits)\n");
    printf("  Max abs error of TF32 vs FP64 ref: %.6e\n", max_abs_err);
    printf("  Max rel error of TF32 vs FP64 ref: %.6e (~%.0f×FP32 ULP)\n",
           max_rel_err, max_rel_err / 1.19e-7);
    printf("  Max ULP diff (TF32 vs FP32 result): %d\n", max_ulp);

    return 0;
}
