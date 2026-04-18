// S3: max power test — mix HMMA + FFMA + STS + LDG in different warps
//
// Hypothesis: cuBLAS BF16 ~2.2 PF saturates tensor pipe but FP32, MIO, LSU
// idle. Adding ops on those pipes should push power higher (toward 1100W TDP).
//
// Design: 1 block of 1024 threads = 32 warps. Each SMSP gets 8 warps:
//   warps 0-1 in each SMSP: HMMA mma.sync (4 chains/warp)
//   warps 2-3: FFMA chain
//   warps 4-5: STS to SMEM
//   warps 6-7: LDG from global (HBM)
//
// 148 blocks (1 per SM). Run for many iters → sustained operation.
// Read power via nvidia-smi externally.
#include <cuda_runtime.h>
#include <cstdio>
#include <unistd.h>

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

__launch_bounds__(1024, 1) __global__ void k_kitchen_sink(int *out, int *gbuf, int N) {
    __shared__ int smem[1024 * 4];
    volatile int *vsmem = smem;
    int slot = (threadIdx.x >> 5) * 32 + (threadIdx.x & 31);
    int warp_id = threadIdx.x >> 5;
    int role = (warp_id >> 1) & 3;  // 0=HMMA, 1=FFMA, 2=STS, 3=LDG
    int v = threadIdx.x;

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
    int la = blockIdx.x * 1024 + threadIdx.x;
    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            if (role == 0) {
                mma_b16(c0, a0, b0, c0);
                mma_b16(c1, a1, b1, c1);
                mma_b16(c2, a2, b2, c2);
                mma_b16(c3, a3, b3, c3);
            } else if (role == 1) {
                fa = fa * fb + fc;  fd = fd * fe + ff;
                fg = fg * fh + fk;  fm = fm * fn + fp;
            } else if (role == 2) {
                vsmem[slot + 0*1024] = v + i + j;
                vsmem[slot + 1*1024] = v + i + j;
                vsmem[slot + 2*1024] = v + i + j;
                vsmem[slot + 3*1024] = v + i + j;
            } else {
                int idx0 = (la + i*4096 + j*32 + 0) % 1000000;
                acc0 ^= gbuf[idx0]; acc1 ^= gbuf[idx0+1];
                acc2 ^= gbuf[idx0+2]; acc3 ^= gbuf[idx0+3];
            }
        }
    }
    if ((c0[0] | (unsigned)fa | (unsigned)acc0) == 0xDEADBEEFu && N < 0)
        out[threadIdx.x] = c0[0] + (int)fa + acc0;
}

__launch_bounds__(1024, 1) __global__ void k_hmma_only(int *out, int *gbuf, int N) {
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
        for (int j = 0; j < N_INNER; j++) {
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
    int *d_data; cudaMalloc(&d_data, 1000004 * sizeof(int));
    cudaMemset(d_data, 0, 1000004 * sizeof(int));
    int N = (argc > 1) ? atoi(argv[1]) : 200;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](const char* name, void(*kfn)(int*,int*,int), int n) {
        for (int i = 0; i < 3; i++) kfn<<<148, 1024>>>(d_out, d_data, n);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) {
            printf("ERR %s\n", cudaGetErrorString(cudaGetLastError()));
            return;
        }
        cudaEventRecord(e0);
        kfn<<<148, 1024>>>(d_out, d_data, n);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        printf("  %-25s N=%d  %.1f ms\n", name, n, ms);
    };
    bench("HMMA only",     k_hmma_only,     N);
    bench("Kitchen sink",  k_kitchen_sink,  N);

    if (argc > 2 && atoi(argv[2]) > 0) {
        int loops = atoi(argv[2]);
        printf("# Sustained %d loops of kitchen_sink with N=%d\n", loops, N);
        for (int i = 0; i < loops; i++) {
            k_kitchen_sink<<<148, 1024>>>(d_out, d_data, N);
        }
        cudaDeviceSynchronize();
    }
    return 0;
}
