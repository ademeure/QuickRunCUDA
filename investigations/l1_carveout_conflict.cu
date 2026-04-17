// l1_carveout_conflict.cu — Diagnose whether elevated latencies at certain WS sizes
// are due to cache-set conflicts (adversarial Sattolo pattern for power-of-2 sizes)
// or are genuine L1 capacity misses.
//
// Method: test with PRIME-sized working sets that break power-of-2 aliasing,
// vs standard power-of-2 sizes. If prime sizes show L1 hit where power-of-2 shows
// elevated latency, it's a conflict-miss artifact.

#include <cstdio>
#include <cuda_runtime.h>

__global__ __launch_bounds__(32, 1)
void pointer_chase_kernel(const int* __restrict__ src,
                          volatile int*            out,
                          int                      n_hops)
{
    if (threadIdx.x != 0) return;
    int idx = 0;
    #pragma unroll 1
    for (int i = 0; i < n_hops; i++) {
        asm volatile("ld.global.ca.s32 %0, [%1];"
            : "=r"(idx) : "l"(src + idx) : "memory");
    }
    *out = idx;
}

static void build_permutation(int* arr, int n, unsigned seed = 0x1337cafe)
{
    for (int i = 0; i < n; i++) arr[i] = i;
    unsigned s = seed;
    for (int i = n - 1; i >= 1; i--) {
        s = s * 1664525u + 1013904223u;
        int j = (int)(s % (unsigned)i);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

static float measure_n(int* d_src, int* d_out, int* h_buf, int n_ints, int carveout)
{
    if (n_ints < 2) n_ints = 2;
    build_permutation(h_buf, n_ints);
    cudaMemcpy(d_src, h_buf, (size_t)n_ints * sizeof(int), cudaMemcpyHostToDevice);

    if (carveout >= 0) {
        cudaFuncSetAttribute((const void*)pointer_chase_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    }

    const int n_hops = 500000;
    for (int w = 0; w < 3; w++)
        pointer_chase_kernel<<<1, 32>>>(d_src, (volatile int*)d_out, n_hops);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float total_ms = 0.0f;
    for (int r = 0; r < 7; r++) {
        cudaEventRecord(t0);
        pointer_chase_kernel<<<1, 32>>>(d_src, (volatile int*)d_out, n_hops);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.0f; cudaEventElapsedTime(&ms, t0, t1);
        total_ms += ms;
    }
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (total_ms / 7.0f * 1e-3f) * (2032.0f * 1e6f) / (float)n_hops;
}

static float measure_kb(int* d_src, int* d_out, int* h_buf, int ws_kb, int carveout)
{
    return measure_n(d_src, d_out, h_buf, ws_kb * 1024 / 4, carveout);
}

// Prime-size helper: round ws_kb to nearest prime number of ints
// Just use n_ints = ws_bytes/4 - 1 if it's odd, or ws_bytes/4 - 3 to avoid power-of-2
static float measure_kb_prime(int* d_src, int* d_out, int* h_buf, int ws_kb, int carveout)
{
    int n = ws_kb * 1024 / 4;
    // Make n odd and not divisible by common factors
    if (n % 2 == 0) n -= 1;
    if (n % 3 == 0) n -= 2;
    return measure_n(d_src, d_out, h_buf, n, carveout);
}

int main()
{
    cudaSetDevice(0);
    printf("GPU: NVIDIA B300 SXM6 AC\n\n");

    int max_bytes = 512 * 1024;
    int* d_src; int* d_out;
    int* h_buf = (int*)malloc((size_t)(max_bytes / sizeof(int)) * sizeof(int));
    cudaMalloc(&d_src, (size_t)(max_bytes / sizeof(int)) * sizeof(int));
    cudaMalloc(&d_out, sizeof(int));

    // --- Test default carveout with power-of-2 vs prime sizes ---
    printf("=== Default carveout: Power-of-2 vs Prime WS sizes ===\n");
    printf("%-10s  %12s  %12s  %s\n", "WS(KB)", "pow2(cy)", "prime(cy)", "verdict");

    int ws_test[] = {32, 48, 64, 80, 96, 112, 128, 160, 192, 256};
    for (int ws_kb : ws_test) {
        float cy_pow2  = measure_kb(d_src, d_out, h_buf, ws_kb, -1);
        float cy_prime = measure_kb_prime(d_src, d_out, h_buf, ws_kb, -1);
        const char* verdict;
        // If prime is much lower than pow2, it's a conflict-miss artifact
        if (cy_prime < cy_pow2 * 0.7f) verdict = "CONFLICT-MISS (pow2 alias)";
        else if (cy_pow2 < 80)          verdict = "L1 hit";
        else if (cy_pow2 > 130)         verdict = "L2 miss";
        else                            verdict = "boundary";
        printf("%-10d  %12.1f  %12.1f  %s\n", ws_kb, cy_pow2, cy_prime, verdict);
    }

    // --- co=0 carveout: same test around the boundary ---
    printf("\n=== co=0 carveout: Power-of-2 vs Prime WS around 192-256 KB ===\n");
    printf("%-10s  %12s  %12s\n", "WS(KB)", "pow2(cy)", "prime(cy)");
    int ws_co0[] = {160, 176, 192, 200, 208, 216, 224, 228, 232, 240};
    for (int ws_kb : ws_co0) {
        float cy_pow2  = measure_kb(d_src, d_out, h_buf, ws_kb, 0);
        float cy_prime = measure_kb_prime(d_src, d_out, h_buf, ws_kb, 0);
        printf("%-10d  %12.1f  %12.1f\n", ws_kb, cy_pow2, cy_prime);
    }

    // --- Final clean reference: co=0 large spacing (safe zones) ---
    printf("\n=== Clean latency reference (co=0, non-aliased sizes) ===\n");
    // Use prime n_ints counts
    int safe_kb[] = {8, 12, 24, 36, 52, 76, 100, 148, 196, 260};
    printf("%-10s  %12s\n", "WS(KB~)", "cy/hop");
    for (int ws_kb : safe_kb) {
        float cy = measure_kb_prime(d_src, d_out, h_buf, ws_kb, 0);
        printf("%-10d  %12.1f\n", ws_kb, cy);
    }

    free(h_buf);
    cudaFree(d_src); cudaFree(d_out);
    return 0;
}
