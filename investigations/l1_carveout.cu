// l1_carveout.cu — L1 cache size characterization across shared memory carveout settings
// B300 sm_103a: unified L1 + SHMEM pool = 256 KB per SM
//
// Method: single-warp pointer-chase (strictly sequential dependent loads).
// One block, one warp. Build a cyclic permutation over 'n_ints' integers so
// the stride walks the entire working set before repeating. Measure clock
// cycles per pointer hop to find the L1→L2 transition.
//
// Carveout semantics (cudaFuncAttributePreferredSharedMemoryCarveout):
//   0   = prefer max L1   (SHMEM carveout fraction = 0%)
//   25  = 25% of pool to SHMEM
//   50  = 50% of pool to SHMEM
//   75  = 75% of pool to SHMEM
//   100 = prefer max SHMEM (L1 carveout fraction = 0%, HW minimum L1 remains)
//
// Compile:
//   nvcc -arch=sm_103a -O3 -o l1_carveout l1_carveout.cu
// Run:
//   ./l1_carveout

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Pointer-chase kernel — strictly serialized, single warp (32 threads but
// only thread 0 does the chase to guarantee one outstanding load at a time).
// We use __launch_bounds__(32, 1) so the compiler knows the block is tiny.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(32, 1)
void pointer_chase_kernel(const int* __restrict__ src,
                          volatile int*            out,
                          int                      n_hops)
{
    if (threadIdx.x != 0) return;

    int idx = 0;
    // Unrolled manually to avoid the compiler killing the chain.
    // We use asm to force sequential dependent LDs.
    #pragma unroll 1
    for (int i = 0; i < n_hops; i++) {
        // Force a dependent load — the compiler cannot hoist or reorder this.
        asm volatile(
            "ld.global.ca.s32 %0, [%1];"
            : "=r"(idx)
            : "l"(src + idx)
            : "memory"
        );
    }
    *out = idx;  // prevent DCE
}

// ---------------------------------------------------------------------------
// Build a random cyclic permutation over [0, n) that visits every element
// exactly once (Sattolo algorithm). Elements are indices (int), so each
// load fetches sizeof(int)=4 bytes.
// ---------------------------------------------------------------------------
static void build_permutation(int* arr, int n, unsigned seed = 0x1337cafe)
{
    // Identity
    for (int i = 0; i < n; i++) arr[i] = i;
    // Sattolo shuffle (single cycle)
    unsigned s = seed;
    for (int i = n - 1; i >= 1; i--) {
        s = s * 1664525u + 1013904223u;  // LCG
        int j = (int)(s % (unsigned)i);  // j in [0, i)
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

// ---------------------------------------------------------------------------
// Measure mean cycles per hop for a given carveout and working-set size.
// Returns cycles/hop (float), or -1 on error.
// ---------------------------------------------------------------------------
static float measure(void* kernel_func,   // pointer_chase_kernel
                     int*  d_src,         // pre-allocated device buffer (max size)
                     int*  d_out,         // 1-int device output
                     int*  h_buf,         // host staging buffer (max size)
                     int   ws_bytes,      // working-set in bytes
                     int   carveout,      // 0..100
                     int   n_warmup,
                     int   n_timed)
{
    int n_ints = ws_bytes / sizeof(int);
    if (n_ints < 2) n_ints = 2;

    // Build permutation on host and upload
    build_permutation(h_buf, n_ints);
    cudaMemcpy(d_src, h_buf, (size_t)n_ints * sizeof(int), cudaMemcpyHostToDevice);

    // Set carveout on the kernel function
    cudaFuncSetAttribute(
        (const void*)kernel_func,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        carveout
    );

    // We want enough hops that timing resolution is good.
    // Target ~500 us at 2032 MHz; L1 hit ~35 cycles → ~14M hops is overkill,
    // use a fixed n_hops that completes in reasonable time even for L2.
    // L2 latency ~200 cycles → 2e6 hops × 200 cy = 4e8 cy = ~200 ms → too slow.
    // Use 1e6 hops for L1-sized WS, fewer for larger.
    // Strategy: scale n_hops so total expected cycles ~50M.
    // Estimate: L1 ~35 cy, L2 ~200 cy. For safety assume 300 cy worst case.
    // 50M / 300 = ~167K hops min. Use 500K.
    const int n_hops = 500000;

    // Warmup
    for (int w = 0; w < n_warmup; w++) {
        pointer_chase_kernel<<<1, 32>>>(d_src, (volatile int*)d_out, n_hops);
    }
    cudaDeviceSynchronize();

    // Timed runs using CUDA events
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

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

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return -1.0f;
    }

    // cycles/hop = (ms / n_timed) * 1e6 [us] * clk_MHz / n_hops
    // With clk = 2032 MHz
    float mean_ms = total_ms / n_timed;
    float clk_MHz = 2032.0f;
    float cycles_per_hop = (mean_ms * 1e-3f) * (clk_MHz * 1e6f) / (float)n_hops;
    return cycles_per_hop;
}

int main()
{
    // Validate GPU
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s  SM%d.%d  SMs=%d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // Working-set sizes in KB → bytes
    // 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 256, 384, 512, 1024, 2048, 4096
    int ws_kb[] = {1,2,4,8,16,32,64,96,128,160,192,256,384,512,1024,2048,4096};
    int n_ws = sizeof(ws_kb) / sizeof(ws_kb[0]);

    // Carveout values
    int carveouts[] = {0, 25, 50, 75, 100};
    int n_co = sizeof(carveouts) / sizeof(carveouts[0]);

    // Allocate max working set on device + host
    int max_bytes = 4096 * 1024;
    int max_ints  = max_bytes / sizeof(int);
    int* d_src;
    int* d_out;
    int* h_buf = (int*)malloc((size_t)max_ints * sizeof(int));
    cudaMalloc(&d_src, (size_t)max_ints * sizeof(int));
    cudaMalloc(&d_out, sizeof(int));

    printf("\n=== L1 Carveout Sweep: cycles/hop (pointer chase, 1 block 1 warp) ===\n");
    printf("GPU clock locked to 2032 MHz\n\n");

    // Print header
    printf("%-10s", "WS");
    for (int c = 0; c < n_co; c++) {
        char hdr[32];
        snprintf(hdr, sizeof(hdr), "C=%d%%", carveouts[c]);
        printf("  %9s", hdr);
    }
    printf("\n");
    printf("%-10s", "(KB)");
    for (int c = 0; c < n_co; c++) printf("  %9s", "---------");
    printf("\n");

    // Store results for analysis
    float results[17][5];  // [ws_idx][co_idx]

    for (int wi = 0; wi < n_ws; wi++) {
        int ws_bytes = ws_kb[wi] * 1024;
        printf("%-10d", ws_kb[wi]);
        fflush(stdout);

        for (int ci = 0; ci < n_co; ci++) {
            // More warmup/timed for small WS (very fast, need timing resolution)
            int n_warmup = 3;
            int n_timed  = 5;

            float cy = measure(
                (void*)pointer_chase_kernel,
                d_src, d_out, h_buf,
                ws_bytes,
                carveouts[ci],
                n_warmup, n_timed
            );
            results[wi][ci] = cy;
            printf("  %9.1f", cy);
            fflush(stdout);
        }
        printf("\n");
    }

    // -----------------------------------------------------------------------
    // Analysis: find L1→L2 transition for each carveout
    // The transition is where latency jumps by > 2× or crosses 100 cycles.
    // -----------------------------------------------------------------------
    printf("\n=== Transition Analysis ===\n");
    printf("(L1 hit: <80 cy,  L2 hit: >120 cy  — rough thresholds)\n\n");

    for (int ci = 0; ci < n_co; ci++) {
        int co = carveouts[ci];
        printf("Carveout=%3d%%: ", co);

        // Find first WS where latency exceeds L1_THRESHOLD
        const float L1_THRESH = 100.0f;
        int transition_kb = -1;
        float l1_lat = 0.0f, l2_lat = 0.0f;
        int n_l1 = 0;

        for (int wi = 0; wi < n_ws; wi++) {
            float cy = results[wi][ci];
            if (cy < L1_THRESH) {
                l1_lat += cy;
                n_l1++;
                transition_kb = ws_kb[wi];  // still in L1
            } else if (l2_lat == 0.0f) {
                l2_lat = cy;
                // The L1 size is the last WS that was in L1
                break;
            }
        }

        if (n_l1 > 0) l1_lat /= n_l1;

        if (transition_kb > 0 && l2_lat > 0.0f) {
            // L1 size is between transition_kb and the next step
            int next_kb = -1;
            for (int wi = 0; wi < n_ws; wi++) {
                if (ws_kb[wi] == transition_kb && wi + 1 < n_ws) {
                    next_kb = ws_kb[wi + 1];
                    break;
                }
            }
            printf("L1 fits up to %4d KB  (L1 lat=%.1f cy, L2 lat=%.1f cy)",
                   transition_kb, l1_lat, l2_lat);
            if (next_kb > 0)
                printf("  → L1 < %d KB", next_kb);
        } else if (transition_kb > 0 && l2_lat == 0.0f) {
            printf("ALL %d KB in L1  (lat=%.1f cy)", ws_kb[n_ws-1], l1_lat);
        } else {
            printf("No clear L1 region found");
        }
        printf("\n");
    }

    // -----------------------------------------------------------------------
    // Print raw data in CSV-friendly format
    // -----------------------------------------------------------------------
    printf("\n=== CSV (for plotting) ===\n");
    printf("ws_kb");
    for (int c = 0; c < n_co; c++) printf(",carveout_%d", carveouts[c]);
    printf("\n");
    for (int wi = 0; wi < n_ws; wi++) {
        printf("%d", ws_kb[wi]);
        for (int ci = 0; ci < n_co; ci++) printf(",%.2f", results[wi][ci]);
        printf("\n");
    }

    // -----------------------------------------------------------------------
    // Query actual SHMEM/L1 config at runtime
    // -----------------------------------------------------------------------
    printf("\n=== CUDA Runtime Cache Config ===\n");
    // Note: cudaFuncGetCacheConfig is deprecated on sm_70+; skip it.
    printf("Cache preference query skipped (deprecated on sm_70+).\n");

    // Query device attribute
    int shared_per_block = 0, shared_per_sm = 0;
    cudaDeviceGetAttribute(&shared_per_block,
        cudaDevAttrMaxSharedMemoryPerBlock, dev);
    cudaDeviceGetAttribute(&shared_per_sm,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    printf("Max SHMEM per block:        %6d bytes = %.0f KB\n",
           shared_per_block, shared_per_block / 1024.0f);
    printf("Max SHMEM per SM:           %6d bytes = %.0f KB\n",
           shared_per_sm, shared_per_sm / 1024.0f);

    // Also query the carveout attribute by trying to read it back
    // (No direct query API; but we can measure what the HW uses.)
    printf("\nNote: B300 unified L1+SHMEM pool = 256 KB per SM (theoretical).\n");
    printf("SHMEM opt-in max = %.0f KB → minimum L1 = %.0f KB at carveout=100.\n",
           shared_per_sm / 1024.0f,
           (256.0f * 1024.0f - shared_per_sm) / 1024.0f);

    free(h_buf);
    cudaFree(d_src);
    cudaFree(d_out);

    return 0;
}
