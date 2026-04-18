// CLEAN: each warp owns IT_COUNT × 1024-B chunks contiguously
// Vary IT_COUNT to make total per-warp bursts longer
#include <cuda_runtime.h>
#include <cstdio>

template<int IT_COUNT>
__launch_bounds__(256, 8) __global__ void w_b(int *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    // Each warp owns IT_COUNT * 1024 bytes (= IT_COUNT * 256 ints)
    int *base = data + warp_id * IT_COUNT * 256;
    int v = 0xab;
    #pragma unroll 1
    for (int it = 0; it < IT_COUNT; it++) {
        int *p = base + (it * 32 + lane) * 8;  // 32 B per lane
        asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
            :: "l"(p), "r"(v) : "memory");
    }
}

int main(int argc, char **argv) {
    int variant = (argc > 1) ? atoi(argv[1]) : 0;
    int it_choices[] = {1, 4, 16, 32, 64, 128};  // 1*1024=1KB, 4=4KB, ... 128=128KB per warp
    int IT = it_choices[variant];

    cudaSetDevice(0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    size_t bytes = 4096ul * 1024 * 1024;
    int *d; cudaMalloc(&d, bytes);

    int per_warp_bytes = IT * 1024;
    int n_warps = bytes / per_warp_bytes;
    int threads = 256;
    int blocks = n_warps / 8;

    auto launch = [&]() {
        switch (variant) {
            case 0: w_b<1  ><<<blocks, threads>>>(d); break;
            case 1: w_b<4  ><<<blocks, threads>>>(d); break;
            case 2: w_b<16 ><<<blocks, threads>>>(d); break;
            case 3: w_b<32 ><<<blocks, threads>>>(d); break;
            case 4: w_b<64 ><<<blocks, threads>>>(d); break;
            case 5: w_b<128><<<blocks, threads>>>(d); break;
        }
    };
    for (int i = 0; i < 3; i++) launch();
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        launch();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    double bw = bytes/(best/1000)/1e9;
    printf("burst/warp=%5d B  blocks=%5d warps=%6d  %.3f ms  %.0f GB/s (%.1f%%)\n",
           per_warp_bytes, blocks, n_warps, best, bw, bw/7672*100);
    return 0;
}
