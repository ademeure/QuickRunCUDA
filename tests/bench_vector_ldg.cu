// Vector LDG width comparison
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int* p = (unsigned int*)A;
    unsigned int* out = (unsigned int*)C;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x);

#if MODE == 0
    // Scalar 32-bit
    unsigned int v = p[idx];
    out[idx] = v + (unsigned)u2;
#elif MODE == 1
    // 64-bit (uint2)
    uint2 v = ((uint2*)p)[idx];
    ((uint2*)out)[idx] = make_uint2(v.x + (unsigned)u2, v.y);
#elif MODE == 2
    // 128-bit (uint4)
    uint4 v = ((uint4*)p)[idx];
    ((uint4*)out)[idx] = make_uint4(v.x + (unsigned)u2, v.y, v.z, v.w);
#elif MODE == 3
    // 256-bit (uint4 + uint4 = 32 bytes via 2 ops)
    uint4 v1 = ((uint4*)p)[idx * 2];
    uint4 v2 = ((uint4*)p)[idx * 2 + 1];
    ((uint4*)out)[idx * 2] = make_uint4(v1.x + (unsigned)u2, v1.y, v1.z, v1.w);
    ((uint4*)out)[idx * 2 + 1] = make_uint4(v2.x + (unsigned)u2, v2.y, v2.z, v2.w);
#endif
}
