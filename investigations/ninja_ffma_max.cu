// NINJA push FFMA past 85.5%: try inline-asm pure FFMA with no loop overhead
// Use HUGE unrolled inner block to minimize branch overhead
#include <cuda_runtime.h>
#include <cstdio>
#include <nvml.h>

extern "C" __launch_bounds__(256, 8) __global__ void ffma_max(float *out, float a_in, int iters) {
    float a = a_in + threadIdx.x * 0.001f;
    float b = a + 1, c = a + 2;
    float r0=0.5f,r1=1.5f,r2=2.5f,r3=3.5f,r4=4.5f,r5=5.5f,r6=6.5f,r7=7.5f;
    float s0=8.5f,s1=9.5f,s2=10.5f,s3=11.5f,s4=12.5f,s5=13.5f,s6=14.5f,s7=15.5f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        // 16 ILP, no extra ops between FFMAs
        // Manually expand to reduce loop count
        #pragma unroll
        for (int k = 0; k < 64; k++) {
            r0=r0*a+b; r1=r1*a+c; r2=r2*a+b; r3=r3*a+c;
            r4=r4*a+b; r5=r5*a+c; r6=r6*a+b; r7=r7*a+c;
            s0=s0*b+a; s1=s1*b+c; s2=s2*b+a; s3=s3*b+c;
            s4=s4*b+a; s5=s5*b+c; s6=s6*b+a; s7=s7*b+c;
        }
    }
    float sum = r0+r1+r2+r3+r4+r5+r6+r7+s0+s1+s2+s3+s4+s5+s6+s7;
    if (sum < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = sum;
}

int main() {
    cudaSetDevice(0);
    nvmlInit();
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex(0, &dev);
    int blocks = 148*8, threads = 256;
    float *d_out; cudaMalloc(&d_out, 1<<24);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 256;
    for (int i = 0; i < 5; i++) ffma_max<<<blocks, threads>>>(d_out, 1.5f, iters);
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(e0);
        ffma_max<<<blocks, threads>>>(d_out, 1.5f, iters);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }

    unsigned mhz; nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz);
    long ffma = (long)blocks * threads * iters * 64 * 16;
    double tflops = ffma * 2.0 / (best/1000) / 1e12;
    double th_at_clk = 148*128*2*(mhz/1000.0)/1000;
    printf("ffma_max blk=%d ILP=16 unroll=64: %.4f ms, %.2f TFLOPS = %.2f%% of %u MHz theoretical (%.1f)\n",
           blocks, best, tflops, tflops/th_at_clk*100, mhz, th_at_clk);
    nvmlShutdown();
    return 0;
}
