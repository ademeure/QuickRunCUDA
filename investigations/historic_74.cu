#include <cuda_runtime.h>
#include <cstdio>

#define KERN(BS, MIN, NAME) \
__launch_bounds__(BS, MIN) __global__ void NAME(float *out, int iters) { \
    float a0=threadIdx.x, a1=a0+1, a2=a0+2, a3=a0+3, a4=a0+4, a5=a0+5, a6=a0+6, a7=a0+7; \
    float b0=a0*2, b1=a1*2, b2=a2*2, b3=a3*2, b4=a4*2, b5=a5*2, b6=a6*2, b7=a7*2; \
    float c0=a0*3, c1=a1*3, c2=a2*3, c3=a3*3, c4=a4*3, c5=a5*3, c6=a6*3, c7=a7*3; \
    for (int i = 0; i < iters; i++) { \
        a0=a0*1.0001f+b0; b0=b0*1.0001f+c0; c0=c0*1.0001f+a0; \
        a1=a1*1.0001f+b1; b1=b1*1.0001f+c1; c1=c1*1.0001f+a1; \
        a2=a2*1.0001f+b2; b2=b2*1.0001f+c2; c2=c2*1.0001f+a2; \
        a3=a3*1.0001f+b3; b3=b3*1.0001f+c3; c3=c3*1.0001f+a3; \
        a4=a4*1.0001f+b4; b4=b4*1.0001f+c4; c4=c4*1.0001f+a4; \
        a5=a5*1.0001f+b5; b5=b5*1.0001f+c5; c5=c5*1.0001f+a5; \
        a6=a6*1.0001f+b6; b6=b6*1.0001f+c6; c6=c6*1.0001f+a6; \
        a7=a7*1.0001f+b7; b7=b7*1.0001f+c7; c7=c7*1.0001f+a7; \
    } \
    float s = a0+a1+a2+a3+a4+a5+a6+a7+b0+b1+b2+b3+b4+b5+b6+b7+c0+c1+c2+c3+c4+c5+c6+c7; \
    if (s < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = s; \
}

KERN(256, 1, k_256_1)
KERN(256, 2, k_256_2)
KERN(256, 4, k_256_4)
KERN(256, 8, k_256_8)

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 148 * 1024 * sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 256;

    auto bench = [&](auto launch, void *fn, int mb) {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, fn);
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 3; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * iters * 24 * 2;
        double tflops = ops / (best/1000.0) / 1e12;
        int max_blk;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, fn, threads, 0);
        printf("  __launch_bounds__(256, %d): regs=%d, blk/SM=%d, %.3f ms = %.1f TFLOPS\n",
               mb, attr.numRegs, max_blk, best, tflops);
    };

    printf("# B300 __launch_bounds__ effect (24 in-flight FMAs, 256 thr/block)\n\n");
    bench([&]{ k_256_1<<<blocks, threads>>>(d_out, iters); }, (void*)k_256_1, 1);
    bench([&]{ k_256_2<<<blocks, threads>>>(d_out, iters); }, (void*)k_256_2, 2);
    bench([&]{ k_256_4<<<blocks, threads>>>(d_out, iters); }, (void*)k_256_4, 4);
    bench([&]{ k_256_8<<<blocks, threads>>>(d_out, iters); }, (void*)k_256_8, 8);

    return 0;
}
