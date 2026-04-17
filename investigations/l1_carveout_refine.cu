// l1_carveout_refine.cu — Refined measurements:
// 1. Default carveout (-1 = driver default)
// 2. Fine-grained WS sweep near boundaries to pin exact L1 sizes
// 3. Explicit latency at hit vs miss to characterize L1 and L2 latencies

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

static float measure_cy(int* d_src, int* d_out, int* h_buf,
                        int ws_bytes, int carveout,
                        int n_warmup, int n_timed)
{
    int n_ints = ws_bytes / sizeof(int);
    if (n_ints < 2) n_ints = 2;
    build_permutation(h_buf, n_ints);
    cudaMemcpy(d_src, h_buf, (size_t)n_ints * sizeof(int), cudaMemcpyHostToDevice);

    if (carveout >= 0) {
        cudaFuncSetAttribute((const void*)pointer_chase_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    }
    // carveout == -1: use driver default (don't call setattr)

    const int n_hops = 500000;
    for (int w = 0; w < n_warmup; w++)
        pointer_chase_kernel<<<1, 32>>>(d_src, (volatile int*)d_out, n_hops);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float total_ms = 0.0f;
    for (int r = 0; r < n_timed; r++) {
        cudaEventRecord(t0);
        pointer_chase_kernel<<<1, 32>>>(d_src, (volatile int*)d_out, n_hops);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        total_ms += ms;
    }
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    float mean_ms = total_ms / n_timed;
    return (mean_ms * 1e-3f) * (2032.0f * 1e6f) / (float)n_hops;
}

int main()
{
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    int max_bytes = 4096 * 1024;
    int* d_src; int* d_out;
    int* h_buf = (int*)malloc((size_t)(max_bytes / sizeof(int)) * sizeof(int));
    cudaMalloc(&d_src, (size_t)(max_bytes / sizeof(int)) * sizeof(int));
    cudaMalloc(&d_out, sizeof(int));

    // --- Test 1: Default carveout ---
    printf("=== Default carveout (no setAttribute call) ===\n");
    printf("%-10s  %9s\n", "WS(KB)", "cycles/hop");
    int ws_coarse[] = {16, 32, 64, 96, 128, 160, 192, 256, 384};
    for (int ws_kb : ws_coarse) {
        float cy = measure_cy(d_src, d_out, h_buf, ws_kb*1024, -1, 3, 5);
        printf("%-10d  %9.1f\n", ws_kb, cy);
    }

    // --- Test 2: Fine sweep near boundaries ---
    // co=0 boundary: between 192 and 256 KB → try 200, 208, 216, 224, 232
    printf("\n=== Fine sweep near co=0 boundary (192-256 KB) ===\n");
    printf("%-10s  %9s\n", "WS(KB)", "cycles/hop");
    int fine0[] = {196, 200, 208, 216, 220, 224, 228, 232, 240, 248, 256};
    for (int ws_kb : fine0) {
        float cy = measure_cy(d_src, d_out, h_buf, ws_kb*1024, 0, 3, 7);
        printf("%-10d  %9.1f\n", ws_kb, cy);
    }

    // co=25 boundary: between 160 and 192 KB (84.7 cy at 192 suggests edge)
    printf("\n=== Fine sweep near co=25 boundary (160-196 KB) ===\n");
    printf("%-10s  %9s\n", "WS(KB)", "cycles/hop");
    int fine25[] = {160, 164, 168, 172, 176, 180, 184, 188, 192, 196};
    for (int ws_kb : fine25) {
        float cy = measure_cy(d_src, d_out, h_buf, ws_kb*1024, 25, 3, 7);
        printf("%-10d  %9.1f\n", ws_kb, cy);
    }

    // co=100 boundary: between 16 and 32 KB → try 20, 24, 28, 32
    printf("\n=== Fine sweep near co=100 boundary (16-32 KB) ===\n");
    printf("%-10s  %9s\n", "WS(KB)", "cycles/hop");
    int fine100[] = {16, 20, 24, 28, 30, 32};
    for (int ws_kb : fine100) {
        float cy = measure_cy(d_src, d_out, h_buf, ws_kb*1024, 100, 3, 7);
        printf("%-10d  %9.1f\n", ws_kb, cy);
    }

    // --- Test 3: co=75 boundary: between 32 and 64 KB → try 40, 48, 56, 60, 64
    printf("\n=== Fine sweep near co=75 boundary (32-64 KB) ===\n");
    printf("%-10s  %9s\n", "WS(KB)", "cycles/hop");
    int fine75[] = {32, 40, 48, 56, 60, 64, 72};
    for (int ws_kb : fine75) {
        float cy = measure_cy(d_src, d_out, h_buf, ws_kb*1024, 75, 3, 7);
        printf("%-10d  %9.1f\n", ws_kb, cy);
    }

    // --- Test 4: L2 latency characterization (co=0, large WS safely in L2)
    printf("\n=== L2 latency (co=0, WS safely above L1) ===\n");
    printf("%-10s  %9s\n", "WS(KB)", "cycles/hop");
    int l2_ws[] = {512, 1024, 2048};
    for (int ws_kb : l2_ws) {
        float cy = measure_cy(d_src, d_out, h_buf, ws_kb*1024, 0, 3, 5);
        printf("%-10d  %9.1f\n", ws_kb, cy);
    }

    free(h_buf);
    cudaFree(d_src); cudaFree(d_out);
    return 0;
}
