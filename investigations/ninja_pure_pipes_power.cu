// C1: power decomposition — pure HBM read vs pure FFMA vs HMMA
// All kernels sustained for several seconds; sample power via nvidia-smi externally.
#include <cuda_runtime.h>
#include <cstdio>

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

__launch_bounds__(512, 1) __global__ void k_ffma_only(int *out, int N) {
    float a0 = threadIdx.x * 1.001f, b0 = 0.999f, c0 = 0.001f;
    float a1 = threadIdx.x * 1.002f, b1 = 0.998f, c1 = 0.002f;
    float a2 = threadIdx.x * 1.003f, b2 = 0.997f, c2 = 0.003f;
    float a3 = threadIdx.x * 1.004f, b3 = 0.996f, c3 = 0.004f;
    float a4 = threadIdx.x * 1.005f, b4 = 0.995f, c4 = 0.005f;
    float a5 = threadIdx.x * 1.006f, b5 = 0.994f, c5 = 0.006f;
    float a6 = threadIdx.x * 1.007f, b6 = 0.993f, c6 = 0.007f;
    float a7 = threadIdx.x * 1.008f, b7 = 0.992f, c7 = 0.008f;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < 64; j++) {
            a0 = a0 * b0 + c0;  a1 = a1 * b1 + c1;
            a2 = a2 * b2 + c2;  a3 = a3 * b3 + c3;
            a4 = a4 * b4 + c4;  a5 = a5 * b5 + c5;
            a6 = a6 * b6 + c6;  a7 = a7 * b7 + c7;
        }
    }
    if (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 == 0.0f && N < 0) out[threadIdx.x] = 1;
}

int main(int argc, char**argv) {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    size_t buf_bytes = 4ull * 1024 * 1024 * 1024;
    int4 *d_data; cudaMalloc(&d_data, buf_bytes);
    cudaMemset(d_data, 0, buf_bytes);
    size_t N_int4 = buf_bytes / 16;
    const char* mode = (argc > 1) ? argv[1] : "hbm";
    int loops = (argc > 2) ? atoi(argv[2]) : 1000;
    if (strcmp(mode, "hbm") == 0) {
        printf("# HBM read sustained: %d loops × 4 GB read\n", loops);
        for (int i = 0; i < loops; i++) k_hbm_read<<<148*8, 256>>>(d_data, d_out, N_int4, 1);
    } else if (strcmp(mode, "ffma") == 0) {
        printf("# FFMA sustained: 8 chains, 512 thr/blk, %d loops\n", loops);
        for (int i = 0; i < loops; i++) k_ffma_only<<<148, 512>>>(d_out, 1000);
    } else {
        printf("Modes: hbm | ffma\n");
        return 1;
    }
    cudaDeviceSynchronize();
    return 0;
}
