// Split SMs between tensor core and FFMA workloads
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void tensor_blocks(float *out, int iters) {
    int laneId = threadIdx.x & 31;
    unsigned a0 = laneId, a1 = laneId+1, a2 = laneId+2, a3 = laneId+3;
    unsigned b0 = laneId*7, b1 = laneId*11;
    float c0=0,c1=0,c2=0,c3=0;

    for (int i = 0; i < iters; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    if (c0+c1+c2+c3 < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = c0+c1+c2+c3;
}

__global__ void ffma_blocks(float *out, int iters) {
    float a = threadIdx.x * 0.001f;
    float b = a + 0.002f;
    float c = b + 0.003f;
    float d = c + 0.004f;
    for (int i = 0; i < iters; i++) {
        a = a*1.0001f + 0.0001f;
        b = b*1.0001f + 0.0001f;
        c = c*1.0001f + 0.0001f;
        d = d*1.0001f + 0.0001f;
    }
    if (a+b+c+d < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s_t, s_f;
    cudaStreamCreateWithFlags(&s_t, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_f, cudaStreamNonBlocking);

    float *d_out; cudaMalloc(&d_out, 148 * 256 * sizeof(float));

    int iters = 100000;

    // Warmup
    tensor_blocks<<<148, 256, 0, s_t>>>(d_out, iters);
    ffma_blocks<<<148, 256, 0, s_f>>>(d_out, iters);
    cudaDeviceSynchronize();

    auto bench = [&](auto fn, int trials = 5) {
        for (int i = 0; i < 3; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("# B300 split-SM tensor + FFMA workloads\n\n");

    // Both with full 148 blocks each (compete for SMs)
    {
        float t = bench([&]{
            tensor_blocks<<<148, 256, 0, s_t>>>(d_out, iters);
            ffma_blocks<<<148, 256, 0, s_f>>>(d_out, iters);
        });
        printf("  Both 148 blocks each (compete):  %.2f ms\n", t);
    }

    // Half SMs each
    {
        float t = bench([&]{
            tensor_blocks<<<74, 256, 0, s_t>>>(d_out, iters);
            ffma_blocks<<<74, 256, 0, s_f>>>(d_out, iters);
        });
        printf("  Both 74 blocks each (split):     %.2f ms\n", t);
    }

    // Tensor alone full
    {
        float t = bench([&]{
            tensor_blocks<<<148, 256, 0, s_t>>>(d_out, iters);
        });
        printf("\n  Tensor alone 148 blocks:         %.2f ms\n", t);
    }
    {
        float t = bench([&]{
            tensor_blocks<<<74, 256, 0, s_t>>>(d_out, iters);
        });
        printf("  Tensor alone 74 blocks:          %.2f ms\n", t);
    }
    {
        float t = bench([&]{
            ffma_blocks<<<148, 256, 0, s_f>>>(d_out, iters);
        });
        printf("\n  FFMA alone 148 blocks:           %.2f ms\n", t);
    }
    {
        float t = bench([&]{
            ffma_blocks<<<74, 256, 0, s_f>>>(d_out, iters);
        });
        printf("  FFMA alone 74 blocks:            %.2f ms\n", t);
    }

    return 0;
}
