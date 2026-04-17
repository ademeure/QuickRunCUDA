// Small (1-256 byte) copy latency comparisons
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void single_byte_writer(char *target, char val) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *target = val;
    }
}

int main() {
    cudaSetDevice(0);

    char *d_buf; cudaMalloc(&d_buf, 4096);
    char h_buf[4096] = {};
    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials = 1000) {
        for (int i = 0; i < 5; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaStreamSynchronize(s);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    printf("# B300 small data transfer methods (write 1 byte to GPU)\n\n");

    // Method 1: cudaMemcpy (synchronous, 1 byte)
    {
        char val = 42;
        float t = bench([&]{
            cudaMemcpy(d_buf, &val, 1, cudaMemcpyHostToDevice);
        });
        printf("  cudaMemcpy (1B sync):           %.2f us\n", t);
    }

    // Method 2: cudaMemcpyAsync (1 byte) + sync
    {
        char val = 42;
        float t = bench([&]{
            cudaMemcpyAsync(d_buf, &val, 1, cudaMemcpyHostToDevice, s);
        });
        printf("  cudaMemcpyAsync (1B + sync):    %.2f us\n", t);
    }

    // Method 3: cudaMemset (1 byte)
    {
        float t = bench([&]{
            cudaMemsetAsync(d_buf, 42, 1, s);
        });
        printf("  cudaMemsetAsync (1B):           %.2f us\n", t);
    }

    // Method 4: launch a kernel that writes 1 byte
    {
        float t = bench([&]{
            single_byte_writer<<<1, 1, 0, s>>>(d_buf, 42);
        });
        printf("  Kernel write (1B):              %.2f us\n", t);
    }

    // Method 5: cuStreamWriteValue32 (4 byte write)
    {
        float t = bench([&]{
            cuStreamWriteValue32(s, (CUdeviceptr)d_buf, 42, 0);
        });
        printf("  cuStreamWriteValue32 (4B):      %.2f us\n", t);
    }

    // Method 6: 1KB copy for comparison
    {
        char val = 42;
        float t = bench([&]{
            cudaMemcpyAsync(d_buf, h_buf, 1024, cudaMemcpyHostToDevice, s);
        });
        printf("\n  cudaMemcpyAsync (1KB+sync):    %.2f us\n", t);
    }

    // Method 7: Read 1 byte back (D2H)
    {
        char val;
        float t = bench([&]{
            cudaMemcpyAsync(&val, d_buf, 1, cudaMemcpyDeviceToHost, s);
        });
        printf("  cudaMemcpyAsync (1B D2H+sync):  %.2f us\n", t);
    }

    return 0;
}
