// Vary warps × chains keeping total HMMA work constant
// If RF read port is the bottleneck, chains=1 + many warps should be fine
// (since each warp has fewer concurrent operand fetches)
#include <cuda_runtime.h>
#include <cstdio>
constexpr int N_INNER = 32;

__device__ __forceinline__ void mma(unsigned (&d)[4], unsigned (&a)[4], unsigned (&b)[2], unsigned (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(c[0]),"r"(c[1]),"r"(c[2]),"r"(c[3]));
}

#define INIT \
    unsigned a0[4]={threadIdx.x|0x3f800001,threadIdx.x|0x3f800002,threadIdx.x|0x3f800003,threadIdx.x|0x3f800004}; \
    unsigned a1[4]={threadIdx.x|0x3f800005,threadIdx.x|0x3f800006,threadIdx.x|0x3f800007,threadIdx.x|0x3f800008}; \
    unsigned a2[4]={threadIdx.x|0x3f80000d,threadIdx.x|0x3f80000e,threadIdx.x|0x3f80000f,threadIdx.x|0x3f800010}; \
    unsigned a3[4]={threadIdx.x|0x3f800011,threadIdx.x|0x3f800012,threadIdx.x|0x3f800013,threadIdx.x|0x3f800014}; \
    unsigned b0[2]={threadIdx.x|0x3f800001,threadIdx.x|0x3f800002}; \
    unsigned b1[2]={threadIdx.x|0x3f800003,threadIdx.x|0x3f800004}; \
    unsigned b2[2]={threadIdx.x|0x3f800005,threadIdx.x|0x3f800006}; \
    unsigned b3[2]={threadIdx.x|0x3f800007,threadIdx.x|0x3f800008}; \
    unsigned c0[4]={0},c1[4]={0},c2[4]={0},c3[4]={0};

template<int THREADS, int CHAINS>
__launch_bounds__(THREADS, 1) __global__ void k(unsigned *out, int N) {
    INIT
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            if (CHAINS >= 1) mma(c0,a0,b0,c0);
            if (CHAINS >= 2) mma(c1,a1,b1,c1);
            if (CHAINS >= 3) mma(c2,a2,b2,c2);
            if (CHAINS >= 4) mma(c3,a3,b3,c3);
        }
    }
    out[blockIdx.x*THREADS + threadIdx.x] = c0[0]+c1[0]+c2[0]+c3[0];
}

int main() {
    cudaSetDevice(0);
    unsigned *out; cudaMalloc(&out, 148*1024*sizeof(unsigned));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto bench = [&](const char* name, auto launch, int warps, int chains, int N) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long total_inst = 148L * warps * (long)N * N_INNER * chains;
        long total_ops = total_inst * 4096;
        double tflops = total_ops / (best/1000.0) / 1e12;
        double mma_per_sm_per_cy = (double)total_inst / 148.0 / (best/1000.0) / 2.032e9;
        printf("  %-30s  %.3f ms  %.1f TF  %.3f mma/SM/cy\n", name, best, tflops, mma_per_sm_per_cy);
    };
    // Keep total work constant: warps × chains × N = 16384 (nominal)
    // Each entry: warps × chains, N adjusted
    bench("4 warps × 4 chains, N=600",  [&](){k<128,4><<<148,128>>>(out, 600);}, 4, 4, 600);
    bench("8 warps × 2 chains, N=600",  [&](){k<256,2><<<148,256>>>(out, 600);}, 8, 2, 600);
    bench("16 warps × 1 chain,  N=600", [&](){k<512,1><<<148,512>>>(out, 600);}, 16, 1, 600);
    bench("16 warps × 2 chains, N=600", [&](){k<512,2><<<148,512>>>(out, 600);}, 16, 2, 600);
    bench("16 warps × 4 chains, N=600", [&](){k<512,4><<<148,512>>>(out, 600);}, 16, 4, 600);
    bench("32 warps × 1 chain,  N=600", [&](){k<1024,1><<<148,1024>>>(out, 600);}, 32, 1, 600);
    bench("32 warps × 4 chains, N=600", [&](){k<1024,4><<<148,1024>>>(out, 600);}, 32, 4, 600);
    return 0;
}
