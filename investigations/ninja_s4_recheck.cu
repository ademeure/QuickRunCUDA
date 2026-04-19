// Re-verify S4: do per-SMSP tensor units exist (linear 1→4 SMSP scaling)?
// Strict anti-DCE this time: thread-derived inputs + always-write
#include <cuda_runtime.h>
#include <cstdio>
constexpr int N_INNER = 64;
__device__ __forceinline__ void mma(unsigned (&d)[4], unsigned (&a)[4], unsigned (&b)[2], unsigned (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(c[0]),"r"(c[1]),"r"(c[2]),"r"(c[3]));
}
template<int THREADS> __launch_bounds__(THREADS, 1) __global__ void k(unsigned *out, int N) {
    unsigned a0[4]={threadIdx.x|0x3f800001,threadIdx.x|0x3f800002,threadIdx.x|0x3f800003,threadIdx.x|0x3f800004};
    unsigned a1[4]={threadIdx.x|0x3f800005,threadIdx.x|0x3f800006,threadIdx.x|0x3f800007,threadIdx.x|0x3f800008};
    unsigned a2[4]={threadIdx.x|0x3f80000d,threadIdx.x|0x3f80000e,threadIdx.x|0x3f80000f,threadIdx.x|0x3f800010};
    unsigned a3[4]={threadIdx.x|0x3f800011,threadIdx.x|0x3f800012,threadIdx.x|0x3f800013,threadIdx.x|0x3f800014};
    unsigned b0[2]={threadIdx.x|0x3f800001,threadIdx.x|0x3f800002};
    unsigned b1[2]={threadIdx.x|0x3f800003,threadIdx.x|0x3f800004};
    unsigned b2[2]={threadIdx.x|0x3f800005,threadIdx.x|0x3f800006};
    unsigned b3[2]={threadIdx.x|0x3f800007,threadIdx.x|0x3f800008};
    unsigned c0[4]={0},c1[4]={0},c2[4]={0},c3[4]={0};
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            mma(c0,a0,b0,c0); mma(c1,a1,b1,c1); mma(c2,a2,b2,c2); mma(c3,a3,b3,c3);
        }
    }
    out[blockIdx.x * THREADS + threadIdx.x] = c0[0]+c1[0]+c2[0]+c3[0];
}
int main() {
    cudaSetDevice(0);
    unsigned *d_out; cudaMalloc(&d_out, 148*1024*sizeof(unsigned));
    int N = 600;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto bench = [&](const char* name, auto launch, int warps_per_sm, int chains_per_warp) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long warps = 148L * warps_per_sm;
        long total_inst = warps * (long)N * N_INNER * chains_per_warp;
        long ops_per_inst = 4096;
        double tflops = total_inst * ops_per_inst / (best/1000.0) / 1e12;
        double mma_per_sm_per_cy = (double)total_inst / 148.0 / (best/1000.0) / 2.032e9;
        printf("  %-30s  %.3f ms  %.1f TF  %.3f mma/SM/cy  (%.1f%% spec)\n",
               name, best, tflops, mma_per_sm_per_cy, tflops/2500*100);
    };
    bench("1 warp/SM   (1 SMSP)",         [&](){k<32><<<148,32>>>(d_out, N);}, 1, 4);
    bench("2 warps/SM  (2 SMSPs)",        [&](){k<64><<<148,64>>>(d_out, N);}, 2, 4);
    bench("4 warps/SM  (4 SMSPs, 1 each)", [&](){k<128><<<148,128>>>(d_out, N);}, 4, 4);
    bench("8 warps/SM  (4 SMSPs, 2 each)", [&](){k<256><<<148,256>>>(d_out, N);}, 8, 4);
    bench("16 warps/SM (4 SMSPs, 4 each)", [&](){k<512><<<148,512>>>(d_out, N);}, 16, 4);
    bench("32 warps/SM (4 SMSPs, 8 each)", [&](){k<1024><<<148,1024>>>(d_out, N);}, 32, 4);
    return 0;
}
