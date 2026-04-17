// cuRAND host API throughput
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    printf("# B300 cuRAND host API throughput\n\n");

    auto bench = [&](curandRngType_t type, const char *name) {
        curandGenerator_t gen;
        curandCreateGenerator(&gen, type);
        curandSetStream(gen, s);
        curandSetPseudoRandomGeneratorSeed(gen, 12345);

        size_t n = 1024 * 1024 * 64;  // 64M floats = 256 MB
        float *d; cudaMalloc(&d, n * sizeof(float));

        // Warm
        for (int i = 0; i < 3; i++) curandGenerateUniform(gen, d, n);
        cudaStreamSynchronize(s);

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0, s);
            curandGenerateUniform(gen, d, n);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double gnums = n / (best/1000.0) / 1e9;
        double gbs = n * 4 / (best/1000.0) / 1e9;
        printf("  %-25s %.3f ms  %.1f Gnum/s  %.1f GB/s\n", name, best, gnums, gbs);

        cudaFree(d);
        curandDestroyGenerator(gen);
    };

    bench(CURAND_RNG_PSEUDO_XORWOW, "XORWOW");
    bench(CURAND_RNG_PSEUDO_MRG32K3A, "MRG32K3A");
    bench(CURAND_RNG_PSEUDO_MTGP32, "MTGP32");
    bench(CURAND_RNG_PSEUDO_PHILOX4_32_10, "Philox4_32_10");
    bench(CURAND_RNG_QUASI_SOBOL32, "Sobol32 (quasi)");

    return 0;
}
