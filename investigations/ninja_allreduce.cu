// A3: 2-GPU all-reduce SoL via NVLink
// Theoretical: bidirectional ring all-reduce on 2 GPUs ≈ 2 × (N/2 GB / 820 GB/s)
// For 1 GB BF16: 2 × 0.5 / 0.820 = 1.22 ms
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <chrono>

constexpr size_t N_BYTES = 1024ull * 1024 * 1024;  // 1 GB

__global__ void k_add(__nv_bfloat16 *dst, const __nv_bfloat16 *src, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < N; i += stride) {
        float a = __bfloat162float(dst[i]);
        float b = __bfloat162float(src[i]);
        dst[i] = __float2bfloat16(a + b);
    }
}

int main() {
    cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1); cudaDeviceEnablePeerAccess(0, 0);

    __nv_bfloat16 *d0, *d1, *d0_recv, *d1_recv;
    cudaSetDevice(0); cudaMalloc(&d0, N_BYTES); cudaMalloc(&d0_recv, N_BYTES / 2);
    cudaSetDevice(1); cudaMalloc(&d1, N_BYTES); cudaMalloc(&d1_recv, N_BYTES / 2);
    cudaSetDevice(0); cudaMemset(d0, 0x3c, N_BYTES);
    cudaSetDevice(1); cudaMemset(d1, 0x3c, N_BYTES);

    size_t N_elems = N_BYTES / 2;
    size_t half_bytes = N_BYTES / 2;
    size_t half_elems = N_elems / 2;

    cudaStream_t s0a, s0b, s1a, s1b;
    cudaSetDevice(0); cudaStreamCreate(&s0a); cudaStreamCreate(&s0b);
    cudaSetDevice(1); cudaStreamCreate(&s1a); cudaStreamCreate(&s1b);
    cudaEvent_t e0, e1;
    cudaSetDevice(0); cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto run_ring = [&]() {
        cudaSetDevice(0);
        cudaMemcpyPeerAsync(d0_recv, 0, d1 + half_elems, 1, half_bytes, s0a);
        cudaSetDevice(1);
        cudaMemcpyPeerAsync(d1_recv, 1, d0, 0, half_bytes, s1a);
        cudaSetDevice(0); cudaStreamSynchronize(s0a);
        k_add<<<148*8, 256, 0, s0a>>>(d0 + half_elems, d0_recv, half_elems);
        cudaSetDevice(1); cudaStreamSynchronize(s1a);
        k_add<<<148*8, 256, 0, s1a>>>(d1, d1_recv, half_elems);
        cudaSetDevice(0); cudaStreamSynchronize(s0a);
        cudaMemcpyPeerAsync(d1 + half_elems, 1, d0 + half_elems, 0, half_bytes, s0b);
        cudaSetDevice(1); cudaStreamSynchronize(s1a);
        cudaMemcpyPeerAsync(d0, 0, d1, 1, half_bytes, s1b);
        cudaSetDevice(0); cudaStreamSynchronize(s0b);
        cudaSetDevice(1); cudaStreamSynchronize(s1b);
    };

    auto bench = [&](const char* name, auto fn, double bytes_per_iter) {
        for (int i = 0; i < 3; i++) fn();
        cudaSetDevice(0); cudaDeviceSynchronize();
        cudaSetDevice(1); cudaDeviceSynchronize();
        double best = 1e30;
        for (int i = 0; i < 5; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaSetDevice(0); cudaDeviceSynchronize();
            cudaSetDevice(1); cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        double bw = bytes_per_iter / (best/1000.0) / 1e9;
        printf("  %-40s  %.3f ms  effective BW = %.0f GB/s\n", name, best, bw);
    };

    printf("# 2× B300 NVLink all-reduce (N = 1 GB BF16, %.0f M elements)\n", (double)N_elems / 1e6);
    printf("# Theoretical: 1.22 ms via bidirectional ring at 820 GB/s\n\n");

    bench("Single-dir P2P copy (0.5 GB)", [&]() {
        cudaSetDevice(0);
        cudaMemcpyPeerAsync(d0_recv, 0, d1, 1, half_bytes, s0a);
        cudaStreamSynchronize(s0a);
    }, half_bytes);

    bench("Bidirectional P2P (0.5 GB ea)", [&]() {
        cudaSetDevice(0);
        cudaMemcpyPeerAsync(d0_recv, 0, d1, 1, half_bytes, s0a);
        cudaSetDevice(1);
        cudaMemcpyPeerAsync(d1_recv, 1, d0, 0, half_bytes, s1a);
        cudaSetDevice(0); cudaStreamSynchronize(s0a);
        cudaSetDevice(1); cudaStreamSynchronize(s1a);
    }, half_bytes * 2);

    bench("Single-dir P2P copy (1 GB)", [&]() {
        cudaSetDevice(0);
        cudaMemcpyPeerAsync(d0, 0, d1, 1, N_BYTES, s0a);
        cudaStreamSynchronize(s0a);
    }, N_BYTES);

    bench("Ring all-reduce (1 GB)", run_ring, N_BYTES);  // 1 GB tensor input

    return 0;
}
