// HBM-saturating + HMMA in different blocks → push close to 940W?
#include <cuda_runtime.h>
#include <cstdio>

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

// HBM grid-stride read (blocks 0..N-1 do this)
__launch_bounds__(256, 8) __global__ void k_hbm_read(const int4 *p, int *out, size_t N, int reps) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    int4 acc = make_int4(0,0,0,0);
    for (int r = 0; r < reps; r++) {
        for (size_t i = tid; i < N; i += stride) {
            int4 v = p[i];
            acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
        }
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0xDEADBEEF && reps < 0)
        out[threadIdx.x] = acc.x;
}

// HMMA pure (different stream, should run concurrently)
__launch_bounds__(512, 1) __global__ void k_hmma_only(int *out, int N) {
    unsigned a0[4] = {0x3f800001, 0x3f800002, 0x3f800003, 0x3f800004};
    unsigned a1[4] = {0x3f800005, 0x3f800006, 0x3f800007, 0x3f800008};
    unsigned a2[4] = {0x3f800009, 0x3f80000a, 0x3f80000b, 0x3f80000c};
    unsigned a3[4] = {0x3f80000d, 0x3f80000e, 0x3f80000f, 0x3f800010};
    unsigned b0[2] = {0x3f800001, 0x3f800002};
    unsigned b1[2] = {0x3f800003, 0x3f800004};
    unsigned b2[2] = {0x3f800005, 0x3f800006};
    unsigned b3[2] = {0x3f800007, 0x3f800008};
    unsigned c0[4] = {0,0,0,0}, c1[4] = {0,0,0,0}, c2[4] = {0,0,0,0}, c3[4] = {0,0,0,0};
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < 64; j++) {
            mma_b16(c0, a0, b0, c0);
            mma_b16(c1, a1, b1, c1);
            mma_b16(c2, a2, b2, c2);
            mma_b16(c3, a3, b3, c3);
        }
    }
    if (c0[0] == 0xDEADBEEFu && N < 0) out[threadIdx.x] = c0[0];
}

int main(int argc, char**argv) {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    size_t buf_bytes = 4ull * 1024 * 1024 * 1024;
    int4 *d_data; cudaMalloc(&d_data, buf_bytes);
    cudaMemset(d_data, 0, buf_bytes);
    size_t N_int4 = buf_bytes / 16;
    cudaStream_t s_hbm, s_hmma;
    cudaStreamCreate(&s_hbm); cudaStreamCreate(&s_hmma);
    int loops = (argc > 1) ? atoi(argv[1]) : 200;
    printf("# HBM + HMMA on separate streams, %d loops\n", loops);
    // Launch HBM stream long enough that HMMA can overlap
    for (int i = 0; i < loops; i++) {
        k_hbm_read<<<148*8, 256, 0, s_hbm>>>(d_data, d_out, N_int4, 1);
        // Launch many HMMA in parallel
        for (int k = 0; k < 5; k++) {
            k_hmma_only<<<148, 512, 0, s_hmma>>>(d_out, 200);
        }
    }
    cudaStreamSynchronize(s_hbm);
    cudaStreamSynchronize(s_hmma);
    return 0;
}
