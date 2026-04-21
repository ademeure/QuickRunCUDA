// R4: Power vs partial-warp activation (predicated lanes)
// Same FFMA loop, but mask off some lanes via predicate
// MODE 0: all 32 lanes active (full warp)
// MODE 1: 16 lanes active (half warp)
// MODE 2: 8 lanes active (quarter)
// MODE 3: 1 lane active
#include <cuda_runtime.h>
#include <cstdio>

__global__ __launch_bounds__(256, 4)
void kernel(float* out, int iters, int u2, int active_lanes) {
    bool active = ((int)threadIdx.x % 32) < active_lanes;
    float a = (float)threadIdx.x * 0.001f + (float)u2 * 1e-9f;
    float b = (float)(threadIdx.x ^ u2) * 0.002f + 1.0f;
    float c = 0.5f;

    for (int i = 0; i < iters; i++) {
#pragma unroll 32
        for (int u = 0; u < 32; u++) {
            // Predicated FFMA via inline PTX
            asm volatile("{ .reg .pred p;\n"
                         "  setp.ne.b32 p, %3, 0;\n"
                         "  @p fma.rn.f32 %0, %0, %1, %2; }"
                         : "+f"(a) : "f"(b), "f"(c), "r"((unsigned)active));
        }
    }

    if ((int)a == 12345) out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

int main(int argc, char** argv) {
    int active_lanes = (argc > 1) ? atoi(argv[1]) : 32;
    int iters = 200000;
    int blocks = 296;
    int threads = 256;
    float* dev_out;
    cudaMalloc(&dev_out, blocks * threads * sizeof(float));

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    kernel<<<blocks, threads>>>(dev_out, 1000, 7, active_lanes);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    kernel<<<blocks, threads>>>(dev_out, iters, 7, active_lanes);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    printf("active_lanes=%d time=%.3f ms\n", active_lanes, ms);
    return 0;
}
