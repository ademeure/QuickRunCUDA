// V5 J2: Subnormal preservation — does B300 FFMA preserve subnormal output bits?
// Build with -ftz=false (no -use_fast_math)
#include <cuda_runtime.h>
#include <cstdio>

__global__ void test_subnormal(unsigned int* out_bits) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // small * small = subnormal output
    // 1.4e-22 * 1.5e-22 = 2.1e-44 (subnormal range, smallest normal ~1.18e-38)
    unsigned int xi = 0x18000001u;  // ~1.65e-22
    unsigned int yi = 0x18000002u;  // similar
    unsigned int zi = 0x00000010u;  // tiny subnormal
    float x = __int_as_float(xi);
    float y = __int_as_float(yi);
    float z = __int_as_float(zi);

    // FFMA (no .ftz with -ftz=false build)
    float r1 = x * y;
    float r2 = x * y + z;

    out_bits[0] = *(unsigned int*)&r1;
    out_bits[1] = *(unsigned int*)&r2;
    // Reference: directly create some subnormals
    unsigned int sub_bits[3] = {0x00000001, 0x00000010, 0x00400000};
    for (int i = 0; i < 3; i++) {
        float f = __int_as_float(sub_bits[i]);
        out_bits[2+i] = *(unsigned int*)&f;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned int* out;
    cudaMallocManaged(&out, 16 * sizeof(unsigned int));
    test_subnormal<<<1, 32>>>(out);
    cudaDeviceSynchronize();
    printf("FFMA with subnormal output (compiled with -ftz=false):\n");
    printf("  x*y      bits: 0x%08x  value: %.6e %s\n", out[0], *(float*)&out[0],
           (out[0] != 0 && (out[0] & 0x7F800000) == 0) ? "(subnormal preserved!)" :
           (out[0] == 0) ? "(FLUSHED to ZERO)" : "(normal)");
    printf("  x*y+z    bits: 0x%08x  value: %.6e %s\n", out[1], *(float*)&out[1],
           (out[1] != 0 && (out[1] & 0x7F800000) == 0) ? "(subnormal preserved!)" :
           (out[1] == 0) ? "(FLUSHED to ZERO)" : "(normal)");
    printf("\nReference subnormals (passed through):\n");
    for (int i = 0; i < 3; i++) {
        printf("  bits 0x%08x → 0x%08x  (preserved: %s)\n",
               (i==0)?0x00000001:(i==1)?0x00000010:0x00400000,
               out[2+i],
               (out[2+i] == ((i==0)?0x00000001:(i==1)?0x00000010:0x00400000)) ? "YES" : "NO (flushed)");
    }
    return 0;
}
