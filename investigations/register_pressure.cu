// Characterize register pressure / spill behavior on B300
// Sweep register usage per thread, measure throughput impact
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define ITERS 20000

// Device function to be inlined
template<int N_REG>
__device__ __forceinline__ void pressure_body(float *out) {
    float regs[N_REG];
    #pragma unroll
    for (int i = 0; i < N_REG; i++) regs[i] = 1.0f + threadIdx.x * 0.001f * (i + 1);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < N_REG; j++) {
            regs[j] = regs[j] * 1.0001f + 0.0001f;
        }
    }

    if (threadIdx.x == 0) {
        float s = 0;
        #pragma unroll
        for (int i = 0; i < N_REG; i++) s += regs[i];
        out[blockIdx.x] = s;
    }
}

extern "C" __global__ void k_4(float *o) { pressure_body<4>(o); }
extern "C" __global__ void k_8(float *o) { pressure_body<8>(o); }
extern "C" __global__ void k_16(float *o) { pressure_body<16>(o); }
extern "C" __global__ void k_32(float *o) { pressure_body<32>(o); }
extern "C" __global__ void k_48(float *o) { pressure_body<48>(o); }
extern "C" __global__ void k_64(float *o) { pressure_body<64>(o); }
extern "C" __global__ void k_96(float *o) { pressure_body<96>(o); }
extern "C" __global__ void k_128(float *o) { pressure_body<128>(o); }

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm = prop.multiProcessorCount;

    float *d_out;
    cudaMalloc(&d_out, sm * sizeof(float));

    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials=5) {
        for (int i = 0; i < 2; i++) { fn(); cudaDeviceSynchronize(); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    auto report = [&](const char *name, void *fn, int N_REG) {
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, fn);

        float t = bench([&]{
            // Call the matching kernel
            if (strcmp(name, "4") == 0) k_4<<<sm, 128, 0, s>>>(d_out);
            else if (strcmp(name, "8") == 0) k_8<<<sm, 128, 0, s>>>(d_out);
            else if (strcmp(name, "16") == 0) k_16<<<sm, 128, 0, s>>>(d_out);
            else if (strcmp(name, "32") == 0) k_32<<<sm, 128, 0, s>>>(d_out);
            else if (strcmp(name, "48") == 0) k_48<<<sm, 128, 0, s>>>(d_out);
            else if (strcmp(name, "64") == 0) k_64<<<sm, 128, 0, s>>>(d_out);
            else if (strcmp(name, "96") == 0) k_96<<<sm, 128, 0, s>>>(d_out);
            else if (strcmp(name, "128") == 0) k_128<<<sm, 128, 0, s>>>(d_out);
        });

        long long ffmas = (long long)sm * 128 * ITERS * N_REG;
        double tflops = (double)(ffmas * 2) / (t/1e3) / 1e12;
        printf("  N_REG=%-3s : regs=%-3d spill=%-4zu time=%.3f ms FFMAs=%lld TFLOPS=%.2f\n",
               name, fa.numRegs, fa.localSizeBytes, t, ffmas, tflops);
    };

    printf("# B300 register pressure sweep\n");
    printf("# 148 blocks × 128 threads, %d iters, N independent FFMA chains per thread\n\n", ITERS);

    report("4", (void*)k_4, 4);
    report("8", (void*)k_8, 8);
    report("16", (void*)k_16, 16);
    report("32", (void*)k_32, 32);
    report("48", (void*)k_48, 48);
    report("64", (void*)k_64, 64);
    report("96", (void*)k_96, 96);
    report("128", (void*)k_128, 128);

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
