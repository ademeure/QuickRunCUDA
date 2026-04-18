// HMMA + FFMA same warp (both compute, no stall) — max power push
#include <cuda_runtime.h>
#include <cstdio>

constexpr int N_INNER = 64;

__device__ __forceinline__ void mma_b16(unsigned (&d)[4],
                                        unsigned (&a)[4], unsigned (&b)[2],
                                        unsigned (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

__launch_bounds__(512, 1) __global__ void k_hmma_ffma(int *out, int N) {
    unsigned a0[4] = {0x3f800001, 0x3f800002, 0x3f800003, 0x3f800004};
    unsigned a1[4] = {0x3f800005, 0x3f800006, 0x3f800007, 0x3f800008};
    unsigned a2[4] = {0x3f800009, 0x3f80000a, 0x3f80000b, 0x3f80000c};
    unsigned a3[4] = {0x3f80000d, 0x3f80000e, 0x3f80000f, 0x3f800010};
    unsigned b0[2] = {0x3f800001, 0x3f800002};
    unsigned b1[2] = {0x3f800003, 0x3f800004};
    unsigned b2[2] = {0x3f800005, 0x3f800006};
    unsigned b3[2] = {0x3f800007, 0x3f800008};
    unsigned c0[4] = {0,0,0,0}, c1[4] = {0,0,0,0}, c2[4] = {0,0,0,0}, c3[4] = {0,0,0,0};
    float fa = threadIdx.x * 1.001f, fb = 0.999f, fc = 0.001f;
    float fd = threadIdx.x * 1.002f, fe = 0.998f, ff = 0.002f;
    float fg = threadIdx.x * 1.003f, fh = 0.997f, fk = 0.003f;
    float fm = threadIdx.x * 1.004f, fn = 0.996f, fp = 0.004f;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            mma_b16(c0, a0, b0, c0);
            fa = fa * fb + fc;  fd = fd * fe + ff;
            mma_b16(c1, a1, b1, c1);
            fg = fg * fh + fk;  fm = fm * fn + fp;
            mma_b16(c2, a2, b2, c2);
            fa = fa * fb + fc;  fd = fd * fe + ff;
            mma_b16(c3, a3, b3, c3);
            fg = fg * fh + fk;  fm = fm * fn + fp;
        }
    }
    if ((c0[0] | (unsigned)fa) == 0xDEADBEEFu && N < 0) out[threadIdx.x] = c0[0] + (int)fa;
}

int main(int argc, char**argv) {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    int N = (argc > 1) ? atoi(argv[1]) : 200;
    int loops = (argc > 2) ? atoi(argv[2]) : 2000;
    printf("# HMMA + FFMA sustained: 512 thr/blk × 148 blocks, N=%d, loops=%d\n", N, loops);
    for (int i = 0; i < loops; i++) {
        k_hmma_ffma<<<148, 512>>>(d_out, N);
    }
    cudaDeviceSynchronize();
    return 0;
}
