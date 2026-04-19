// Find ACTUAL mma.sync BF16 peak with strict anti-DCE
// Catalog claims 577 TF at ILP=2, 64w/SM — verify
#include <cuda_runtime.h>
#include <cstdio>
constexpr int N_INNER = 32;

__device__ __forceinline__ void mma(unsigned (&d)[4], unsigned (&a)[4], unsigned (&b)[2], unsigned (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(c[0]),"r"(c[1]),"r"(c[2]),"r"(c[3]));
}

#define INIT_AB \
    unsigned a0[4]={threadIdx.x|0x3f800001,threadIdx.x|0x3f800002,threadIdx.x|0x3f800003,threadIdx.x|0x3f800004}; \
    unsigned a1[4]={threadIdx.x|0x3f800005,threadIdx.x|0x3f800006,threadIdx.x|0x3f800007,threadIdx.x|0x3f800008}; \
    unsigned a2[4]={threadIdx.x|0x3f80000d,threadIdx.x|0x3f80000e,threadIdx.x|0x3f80000f,threadIdx.x|0x3f800010}; \
    unsigned a3[4]={threadIdx.x|0x3f800011,threadIdx.x|0x3f800012,threadIdx.x|0x3f800013,threadIdx.x|0x3f800014}; \
    unsigned b0[2]={threadIdx.x|0x3f800001,threadIdx.x|0x3f800002}; \
    unsigned b1[2]={threadIdx.x|0x3f800003,threadIdx.x|0x3f800004}; \
    unsigned b2[2]={threadIdx.x|0x3f800005,threadIdx.x|0x3f800006}; \
    unsigned b3[2]={threadIdx.x|0x3f800007,threadIdx.x|0x3f800008};

template<int THREADS, int MIN_BLOCKS, int CHAINS>
__launch_bounds__(THREADS, MIN_BLOCKS) __global__ void k(unsigned *out, int N) {
    INIT_AB
    unsigned c0[4]={0},c1[4]={0},c2[4]={0},c3[4]={0};
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            if (CHAINS >= 1) mma(c0,a0,b0,c0);
            if (CHAINS >= 2) mma(c1,a1,b1,c1);
            if (CHAINS >= 3) mma(c2,a2,b2,c2);
            if (CHAINS >= 4) mma(c3,a3,b3,c3);
        }
    }
    out[blockIdx.x * THREADS + threadIdx.x] = c0[0]+c1[0]+c2[0]+c3[0];
}

int main() {
    cudaSetDevice(0);
    unsigned *out; cudaMalloc(&out, 148*1024*8*sizeof(unsigned));
    int N = 600;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto bench = [&](const char* name, auto launch, int blocks_per_sm, int warps_per_block, int chains) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR\n"); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long warps = 148L * blocks_per_sm * warps_per_block;
        long total_ops = warps * (long)N * N_INNER * chains * 4096;
        double tflops = total_ops / (best/1000.0) / 1e12;
        printf("  %-30s  %.3f ms  %.1f TF (%.1f%% of 616)\n",
               name, best, tflops, tflops/616*100);
    };
    // Sweep ILP × warps/SM
    bench("ILP=4, 16w/SM (8w*2blk)", [&](){k<256, 2, 4><<<148*2, 256>>>(out, N);}, 2, 8, 4);
    bench("ILP=2, 16w/SM (8w*2blk)", [&](){k<256, 2, 2><<<148*2, 256>>>(out, N);}, 2, 8, 2);
    bench("ILP=2, 32w/SM (16w*2blk)",[&](){k<512, 2, 2><<<148*2, 512>>>(out, N);}, 2, 16, 2);
    bench("ILP=2, 64w/SM (32w*2blk)",[&](){k<1024, 2, 2><<<148*2, 1024>>>(out, N);}, 2, 32, 2);
    bench("ILP=4, 32w/SM (16w*2blk)",[&](){k<512, 2, 4><<<148*2, 512>>>(out, N);}, 2, 16, 4);
    bench("ILP=1, 64w/SM (32w*2blk)",[&](){k<1024, 2, 1><<<148*2, 1024>>>(out, N);}, 2, 32, 1);
    return 0;
}
