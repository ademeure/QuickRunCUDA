// 3D memcpy vs flat memcpy
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);

    auto bench = [&](auto fn, int trials = 5) {
        for (int i = 0; i < 2; i++) fn();
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

    cudaStream_t s; cudaStreamCreate(&s);

    printf("# B300 cudaMemcpy3DAsync vs flat cudaMemcpyAsync (D2D)\n\n");

    for (int D : {16, 64, 256, 1024}) {
        size_t total = (size_t)D * D * D;  // elements
        size_t bytes = total * sizeof(float);
        if (bytes > 4ull*1024*1024*1024) continue;

        float *d_src; cudaMalloc(&d_src, bytes);
        float *d_dst; cudaMalloc(&d_dst, bytes);
        cudaMemset(d_src, 0, bytes);

        // Flat copy
        float t_flat = bench([&]{
            cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, s);
        });

        // 3D copy
        cudaMemcpy3DParms p = {};
        p.srcPtr = make_cudaPitchedPtr(d_src, D*sizeof(float), D, D);
        p.dstPtr = make_cudaPitchedPtr(d_dst, D*sizeof(float), D, D);
        p.extent = make_cudaExtent(D*sizeof(float), D, D);
        p.kind = cudaMemcpyDeviceToDevice;

        float t_3d = bench([&]{
            cudaMemcpy3DAsync(&p, s);
        });

        double bw_flat = bytes / (t_flat/1000) / 1e9;
        double bw_3d = bytes / (t_3d/1000) / 1e9;
        printf("  %d³ floats (%6.1f MB): flat=%.2f ms (%6.1f GB/s) 3d=%.2f ms (%6.1f GB/s)\n",
               D, bytes/1024.0/1024, t_flat, bw_flat, t_3d, bw_3d);

        cudaFree(d_src); cudaFree(d_dst);
    }

    return 0;
}
