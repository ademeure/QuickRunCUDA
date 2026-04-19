// HMMA sustained power with STRICT anti-DCE (post S1 retraction)
#include <cuda_runtime.h>
#include <cstdio>
constexpr int N_INNER = 64;
__device__ __forceinline__ void mma(unsigned (&d)[4], unsigned (&a)[4], unsigned (&b)[2], unsigned (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(c[0]),"r"(c[1]),"r"(c[2]),"r"(c[3]));
}
__launch_bounds__(256, 1) __global__ void k_strict(unsigned *out, int N) {
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
    out[blockIdx.x * 256 + threadIdx.x] = c0[0]+c1[0]+c2[0]+c3[0];
}
int main(int argc, char**argv) {
    cudaSetDevice(0);
    unsigned *d_out; cudaMalloc(&d_out, 148*256*sizeof(unsigned));
    int N = (argc > 1) ? atoi(argv[1]) : 200;
    int loops = (argc > 2) ? atoi(argv[2]) : 5000;
    printf("# HMMA strict anti-DCE: 8 warps/SM (256 thr/blk × 148 blocks), N=%d, %d loops\n", N, loops);
    for (int i = 0; i < loops; i++) k_strict<<<148, 256>>>(d_out, N);
    cudaDeviceSynchronize();
    return 0;
}
