// Check power under tensor core (mma.sync) load
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <chrono>

__global__ void tensor_hot(float *out, int iters) {
    int laneId = threadIdx.x & 31;
    unsigned a0 = laneId, a1 = laneId+1, a2 = laneId+2, a3 = laneId+3;
    unsigned b0 = laneId*7, b1 = laneId*11;
    float c0=0,c1=0,c2=0,c3=0;
    float d0=0,d1=0,d2=0,d3=0;

    for (int i = 0; i < iters; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
            : "r"(a1),"r"(a2),"r"(a3),"r"(a0),"r"(b1),"r"(b0));
    }
    if (c0+c1+c2+c3+d0+d1+d2+d3 < -1e30f)
        out[blockIdx.x*blockDim.x+threadIdx.x] = c0+c1+c2+c3+d0+d1+d2+d3;
}

int main() {
    cudaSetDevice(0);
    nvmlInit_v2();
    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex_v2(0, &dev);

    float *d_out; cudaMalloc(&d_out, 148*256*sizeof(float));

    printf("# B300 power under tensor core (BF16 mma.sync ILP=2) sustained load\n\n");
    printf("# %-8s %-12s %-10s %-10s %-12s\n",
           "t_s", "elapsed_ms", "Power_W", "Temp_C", "TFLOPS");

    auto t_start = std::chrono::high_resolution_clock::now();
    while (true) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int i = 0; i < 5; i++) tensor_hot<<<148, 256>>>(d_out, 100000);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);

        int warps = 148 * 8;
        long ops = 5L * warps * 100000 * 2 * 4096;  // 2 ILP × 4096 ops
        double tflops = ops / (ms/1000.0) / 1e12;

        unsigned int pw, t_c;
        nvmlDeviceGetPowerUsage(dev, &pw);
        nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &t_c);

        auto t_now = std::chrono::high_resolution_clock::now();
        float t_s = std::chrono::duration<float>(t_now - t_start).count();
        printf("  %-8.1f %-12.2f %-10.1f %-10u %-12.0f\n", t_s, ms, pw/1000.0, t_c, tflops);

        cudaEventDestroy(e0); cudaEventDestroy(e1);
        if (t_s > 15) break;
    }

    nvmlShutdown();
    return 0;
}
