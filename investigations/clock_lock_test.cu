// Verify FFMA TFLOPS at locked clocks vs default boost
#include <unistd.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void ffma(float *out, int iters, float k1, float k2) {
    float a = threadIdx.x * 0.001f;
    float b = threadIdx.x * 0.002f;
    float c = threadIdx.x * 0.003f;
    float d = threadIdx.x * 0.004f;
    for (int i = 0; i < iters; i++) {
        a = a*k1 + k2;
        b = b*k1 + k2;
        c = c*k1 + k2;
        d = d*k1 + k2;
    }
    if (a+b+c+d < -1e30f) out[blockIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 1024 * sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 128;
    long total_flops = (long)blocks * threads * iters * 4 * 2;  // 4 chains × FMA = 2 FLOPs

    auto bench = [&]() {
        for (int i = 0; i < 3; i++) ffma<<<blocks, threads>>>(d_out, iters, 1.0001f, 0.0001f);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            ffma<<<blocks, threads>>>(d_out, iters, 1.0001f, 0.0001f);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return total_flops / (best/1000.0) / 1e12;  // TFLOPS
    };

    printf("# B300 FFMA TFLOPS at various clock states\n");

    // Get current clock
    auto get_clk = [](){
        char buf[256];
        FILE *p = popen("nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader -i 0 | tr -d '\\n MHz'", "r");
        fgets(buf, sizeof(buf), p); pclose(p);
        return atoi(buf);
    };

    // Baseline (default boost)
    printf("  Default boost: clock=%d MHz, %.1f TFLOPS\n", get_clk(), bench());

    // Lock to 2032 (paradoxically becomes 1920)
    system("nvidia-smi -lgc 2032 -i 0 > /dev/null");
    sleep(1);
    printf("  -lgc 2032:     clock=%d MHz, %.1f TFLOPS\n", get_clk(), bench());

    // Lock to 1410 (lower clock)
    system("nvidia-smi -lgc 1410 -i 0 > /dev/null");
    sleep(1);
    printf("  -lgc 1410:     clock=%d MHz, %.1f TFLOPS\n", get_clk(), bench());

    // Reset
    system("nvidia-smi -rgc -i 0 > /dev/null");
    sleep(1);
    printf("  -rgc (reset):  clock=%d MHz, %.1f TFLOPS\n", get_clk(), bench());

    return 0;
}
