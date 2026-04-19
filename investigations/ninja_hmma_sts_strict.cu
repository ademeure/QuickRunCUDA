// HMMA+STS combo with strict anti-DCE
#include <cuda_runtime.h>
#include <cstdio>
constexpr int N_INNER = 64;
__device__ __forceinline__ void mma(unsigned (&d)[4], unsigned (&a)[4], unsigned (&b)[2], unsigned (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(c[0]),"r"(c[1]),"r"(c[2]),"r"(c[3]));
}

template<int H, int S>
__launch_bounds__(512, 1) __global__ void k(unsigned *out, int N) {
    __shared__ int smem[1024 * 4];
    volatile int *vsmem = smem;
    int slot = (threadIdx.x >> 5) * 32 + (threadIdx.x & 31);
    int v = threadIdx.x;
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
            if (H >= 1) mma(c0,a0,b0,c0);
            if (S >= 1) vsmem[slot+0*1024] = v+i;
            if (H >= 2) mma(c1,a1,b1,c1);
            if (S >= 2) vsmem[slot+1*1024] = v+i;
            if (H >= 3) mma(c2,a2,b2,c2);
            if (S >= 3) vsmem[slot+2*1024] = v+i;
            if (H >= 4) mma(c3,a3,b3,c3);
            if (S >= 4) vsmem[slot+3*1024] = v+i;
        }
    }
    out[blockIdx.x * 512 + threadIdx.x] = c0[0]+c1[0]+c2[0]+c3[0];
}

int main() {
    cudaSetDevice(0);
    unsigned *out; cudaMalloc(&out, 148*512*sizeof(unsigned));
    int N = 200;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto bench = [&](const char* name, auto launch) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        printf("  %-25s  %.3f ms\n", name, best);
    };
    printf("# HMMA+STS combo strict anti-DCE\n");
    bench("HMMA=1, STS=0", [&](){k<1,0><<<148, 512>>>(out, N);});
    bench("HMMA=4, STS=0", [&](){k<4,0><<<148, 512>>>(out, N);});
    bench("HMMA=0, STS=1", [&](){k<0,1><<<148, 512>>>(out, N);});
    bench("HMMA=0, STS=4", [&](){k<0,4><<<148, 512>>>(out, N);});
    bench("HMMA=1, STS=1", [&](){k<1,1><<<148, 512>>>(out, N);});
    bench("HMMA=4, STS=1", [&](){k<4,1><<<148, 512>>>(out, N);});
    bench("HMMA=1, STS=4", [&](){k<1,4><<<148, 512>>>(out, N);});
    bench("HMMA=4, STS=4", [&](){k<4,4><<<148, 512>>>(out, N);});
    return 0;
}
