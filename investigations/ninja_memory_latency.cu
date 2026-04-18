#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void chase_init(uint64_t *p, uint64_t N) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    p[tid] = (tid * 65537ULL + 7919ULL) % N;
}

// Warm L2 by reading entire range
__global__ void warm(const uint64_t *p, uint64_t N, uint64_t *sink) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t acc = 0;
    for (uint64_t i = tid; i < N; i += gridDim.x * blockDim.x) acc ^= p[i];
    if (acc == 0xdeadbeefULL) sink[tid] = acc;
}

__global__ void chase(const uint64_t *p, uint64_t *out, int N_loads) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint64_t cur = 0;
    long long t0 = clock64();
    for (int i = 0; i < N_loads; i++) cur = p[cur];
    long long t1 = clock64();
    out[0] = cur;
    out[1] = (uint64_t)(t1 - t0);
}

int main() {
    cudaSetDevice(0);
    uint64_t Nmax = 256ULL * 1024 * 1024;
    uint64_t *d_p; cudaMalloc(&d_p, Nmax * 8);
    uint64_t *d_out; cudaMalloc(&d_out, 64);
    uint64_t *d_sink; cudaMalloc(&d_sink, 1024 * 1024);
    uint64_t h[2];
    int N_loads = 200;

    auto run = [&](uint64_t N, const char* name, bool warm_l2) {
        chase_init<<<(N+255)/256, 256>>>(d_p, N);
        cudaDeviceSynchronize();
        if (warm_l2) {
            // Read entire range so it's in L2
            warm<<<148, 256>>>(d_p, N, d_sink);
            cudaDeviceSynchronize();
        }
        // Run 1: warmup (may be cold)
        chase<<<1, 32>>>(d_p, d_out, N_loads);
        cudaDeviceSynchronize();
        // Run 2: measure
        chase<<<1, 32>>>(d_p, d_out, N_loads);
        cudaDeviceSynchronize();
        cudaMemcpy(h, d_out, 16, cudaMemcpyDeviceToHost);
        double per_cy = (double)h[1] / N_loads;
        printf("  %-30s WS=%4llu KB  %5.0f cy = %5.1f ns\n", name, N*8/1024,
               per_cy, per_cy / 2.032);
    };

    // L1 territory
    run(256, "L1 chase (2 KB WS)", false);
    // L2 territory
    run(64 * 1024, "L2 chase (512 KB WS, warmed)", true);
    run(2 * 1024 * 1024, "L2 chase (16 MB WS, warmed)", true);
    run(4 * 1024 * 1024, "L2 chase (32 MB WS, warmed)", true);
    // DRAM
    run(8 * 1024 * 1024, "DRAM chase (64 MB > L2)", false);
    run(64 * 1024 * 1024, "DRAM chase (512 MB)", false);
    run(256 * 1024 * 1024, "DRAM chase (2 GB)", false);
    return 0;
}
