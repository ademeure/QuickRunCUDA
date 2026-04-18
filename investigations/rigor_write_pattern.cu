// Test: coalesced consecutive writes vs strided
#include <cuda_runtime.h>
#include <cstdio>

// Stride pattern (current 8-ILP test): each warp writes to far-apart lines
__launch_bounds__(512, 4) __global__ void write_strided(int4 *data, int N, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int4 v = make_int4(seed, seed+1, seed+2, seed+3);
    for (int i = tid; i < N - 7*stride; i += 8*stride) {
        data[i] = v;
        data[i + stride] = v;
        data[i + 2*stride] = v;
        data[i + 3*stride] = v;
        data[i + 4*stride] = v;
        data[i + 5*stride] = v;
        data[i + 6*stride] = v;
        data[i + 7*stride] = v;
    }
}

// Coalesced consecutive: each warp writes to 8 consecutive cache lines
__launch_bounds__(512, 4) __global__ void write_consec(int4 *data, int N, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int4 v = make_int4(seed, seed+1, seed+2, seed+3);
    // Each thread writes 8 int4s in CONSECUTIVE positions, then jumps by 8*stride
    for (int base = tid * 8; base < N - 7; base += stride * 8) {
        data[base + 0] = v;
        data[base + 1] = v;
        data[base + 2] = v;
        data[base + 3] = v;
        data[base + 4] = v;
        data[base + 5] = v;
        data[base + 6] = v;
        data[base + 7] = v;
    }
}

int main() {
    cudaSetDevice(0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    size_t bytes = 4096ul * 1024 * 1024;
    int N = bytes / 16;
    int4 *d; cudaMalloc(&d, bytes);

    auto bench = [&](auto launch) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 7; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("# Write pattern audit: strided vs coalesced consecutive\n\n");
    printf("# %-25s %-12s %-12s %-12s\n", "method", "ms", "GB/s", "%peak");

    {
        float t = bench([&]{ write_strided<<<148, 512>>>(d, N, 1); });
        double bw = bytes/(t/1000)/1e9;
        printf("  %-25s %-12.3f %-12.0f %-12.1f\n", "strided 8-ILP (orig)", t, bw, bw/7672*100);
    }
    {
        float t = bench([&]{ write_consec<<<148, 512>>>(d, N, 1); });
        double bw = bytes/(t/1000)/1e9;
        printf("  %-25s %-12.3f %-12.0f %-12.1f\n", "consec 8-ILP", t, bw, bw/7672*100);
    }
    {
        float t = bench([&]{ cudaMemsetAsync(d, 0xab, bytes, 0); });
        double bw = bytes/(t/1000)/1e9;
        printf("  %-25s %-12.3f %-12.0f %-12.1f\n", "cudaMemset (ref)", t, bw, bw/7672*100);
    }

    return 0;
}
