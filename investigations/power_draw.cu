// Observe B300 power draw during different workloads
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <thread>

extern "C" __global__ void ffma_stress(float *out) {
    float a = 1.0f + threadIdx.x * 0.001f;
    float b = 1.00001f, c = 0.00001f;
    // Run for a long time
    #pragma unroll 1
    for (int i = 0; i < 2000000; i++) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(b), "f"(c));
        }
    }
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void dram_stress(float *in, float *out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float sum = 0;
    for (int i = 0; i < 2000; i++) {
        for (int j = tid; j < N; j += stride) sum += in[j];
    }
    if (sum < -1e30f) out[tid] = sum;
}

extern "C" __global__ void noop_stress(int *out) {
    int i = 0;
    for (int k = 0; k < 10000000; k++) {
        asm volatile("add.u32 %0, %0, 1;" : "+r"(i));
    }
    if (threadIdx.x == 0) out[blockIdx.x] = i;
}

int main() {
    cudaSetDevice(0);

    float *d_out; cudaMalloc(&d_out, 1024 * sizeof(float));
    float *d_in; cudaMalloc(&d_in, 256 * 1024 * 1024);
    cudaMemset(d_in, 0x40, 256 * 1024 * 1024);
    int *d_noop; cudaMalloc(&d_noop, 1024 * sizeof(int));

    cudaStream_t s; cudaStreamCreate(&s);

    printf("# B300 workload characterization\n");
    printf("# Will run each workload for ~5 sec, you can observe via nvidia-smi\n\n");

    // Warm up first
    ffma_stress<<<148, 256, 0, s>>>(d_out);
    cudaDeviceSynchronize();

    auto run_workload = [&](const char *name, auto fn) {
        printf("=== Running %s ===\n", name);
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        float s_time = std::chrono::duration<float>(t1-t0).count();
        printf("  duration: %.2f s\n", s_time);
        // Sample nvidia-smi
        system("nvidia-smi --query-gpu=power.draw,clocks.current.sm,temperature.gpu,utilization.gpu --format=csv,noheader | head -2");
    };

    // FFMA stress
    run_workload("FFMA compute stress", [&]{
        for (int rep = 0; rep < 20; rep++) {
            ffma_stress<<<296, 1024, 0, s>>>(d_out);
        }
    });

    // DRAM stress
    run_workload("DRAM read stress", [&]{
        dram_stress<<<296, 512, 0, s>>>(d_in, d_out, 64 * 1024 * 1024);
    });

    // NOP stress (idle-ish)
    run_workload("NOP-only stress", [&]{
        for (int i = 0; i < 20; i++) {
            noop_stress<<<148, 32, 0, s>>>(d_noop);
        }
    });

    printf("\n# Final state after all tests:\n");
    system("nvidia-smi --query-gpu=power.draw,clocks.current.sm,temperature.gpu,utilization.gpu --format=csv,noheader | head -2");

    cudaFree(d_out); cudaFree(d_in); cudaFree(d_noop);
    return 0;
}
