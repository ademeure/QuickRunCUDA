// In-kernel clock64() measurement for 32x4 mystery
#include <cuda_runtime.h>
#include <cstdio>
constexpr int N_INNER = 32;
__device__ __forceinline__ void mma(unsigned (&d)[4], unsigned (&a)[4], unsigned (&b)[2], unsigned (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(c[0]),"r"(c[1]),"r"(c[2]),"r"(c[3]));
}
template<int THREADS>
__launch_bounds__(THREADS, 1) __global__ void k_clk(unsigned *out, unsigned long long *clk_out, int N) {
    unsigned a0[4]={threadIdx.x|0x3f800001,threadIdx.x|0x3f800002,threadIdx.x|0x3f800003,threadIdx.x|0x3f800004};
    unsigned a1[4]={threadIdx.x|0x3f800005,threadIdx.x|0x3f800006,threadIdx.x|0x3f800007,threadIdx.x|0x3f800008};
    unsigned a2[4]={threadIdx.x|0x3f80000d,threadIdx.x|0x3f80000e,threadIdx.x|0x3f80000f,threadIdx.x|0x3f800010};
    unsigned a3[4]={threadIdx.x|0x3f800011,threadIdx.x|0x3f800012,threadIdx.x|0x3f800013,threadIdx.x|0x3f800014};
    unsigned b0[2]={threadIdx.x|0x3f800001,threadIdx.x|0x3f800002};
    unsigned b1[2]={threadIdx.x|0x3f800003,threadIdx.x|0x3f800004};
    unsigned b2[2]={threadIdx.x|0x3f800005,threadIdx.x|0x3f800006};
    unsigned b3[2]={threadIdx.x|0x3f800007,threadIdx.x|0x3f800008};
    unsigned c0[4]={0},c1[4]={0},c2[4]={0},c3[4]={0};
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            mma(c0,a0,b0,c0); mma(c1,a1,b1,c1); mma(c2,a2,b2,c2); mma(c3,a3,b3,c3);
        }
    }
    unsigned long long t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[blockIdx.x*THREADS + threadIdx.x] = c0[0]+c1[0]+c2[0]+c3[0];
    if (threadIdx.x == 0) clk_out[blockIdx.x] = t1 - t0;
}

int main() {
    cudaSetDevice(0);
    unsigned *out; cudaMalloc(&out, 148*1024*sizeof(unsigned));
    unsigned long long *clk_out; cudaMalloc(&clk_out, 148*sizeof(unsigned long long));
    int N = 600;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto bench = [&](const char* name, auto launch, int threads) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        unsigned long long h_clk[148]; cudaMemcpy(h_clk, clk_out, 148*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        unsigned long long min_cy = h_clk[0], max_cy = h_clk[0], sum = 0;
        for (int i = 0; i < 148; i++) { sum += h_clk[i]; if (h_clk[i]<min_cy) min_cy=h_clk[i]; if (h_clk[i]>max_cy) max_cy=h_clk[i]; }
        double avg_cy = (double)sum/148;
        double clk_ms = max_cy / 2.032e6;
        printf("  %-25s  wall=%.3f ms  clk64 max=%llu (%.3f ms) min=%llu avg=%.0f\n",
               name, best, max_cy, clk_ms, min_cy, avg_cy);
    };
    bench("16 warps × 4 chains", [&](){k_clk<512><<<148, 512>>>(out, clk_out, N);}, 512);
    bench("32 warps × 4 chains", [&](){k_clk<1024><<<148, 1024>>>(out, clk_out, N);}, 1024);
    return 0;
}
