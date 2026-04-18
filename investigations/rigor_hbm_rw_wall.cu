// Measure WALL-CLOCK from earliest-start to latest-end of concurrent R+W
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void w_v8(int *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *base = data + warp_id * (32 * 1024 / 4);
    int v = 0xab;
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        int *p = base + (it * 32 + lane) * 8;
        asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
            :: "l"(p), "r"(v) : "memory");
    }
}

__launch_bounds__(256, 8) __global__ void r_v8(int *data, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *base = data + warp_id * (32 * 1024 / 4);
    int acc = 0;
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        int *p = base + (it * 32 + lane) * 8;
        int r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "l"(p));
        acc ^= r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7;
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s_r, s_w;
    cudaStreamCreateWithFlags(&s_r, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_w, cudaStreamNonBlocking);
    // Event on default stream synchronizes with BOTH
    cudaEvent_t global_start, global_end;
    cudaEventCreate(&global_start); cudaEventCreate(&global_end);

    size_t bytes = 4096ul * 1024 * 1024;
    int *d_a; cudaMalloc(&d_a, bytes); cudaMemset(d_a, 0xab, bytes);
    int *d_b; cudaMalloc(&d_b, bytes);
    int *d_out; cudaMalloc(&d_out, 16384 * 256 * sizeof(int));
    int blocks = bytes / (256 * 1024), threads = 256;

    for (int i = 0; i < 3; i++) {
        r_v8<<<blocks, threads, 0, s_r>>>(d_a, d_out);
        w_v8<<<blocks, threads, 0, s_w>>>(d_b);
    }
    cudaDeviceSynchronize();

    // Global wall-clock = max(r_end, w_end) - min(r_start, w_start)
    // Easiest: sync everything before, launch both on non-blocking, then sync and time
    float best_total = 1e30f;
    for (int trial = 0; trial < 5; trial++) {
        cudaDeviceSynchronize();
        cudaEventRecord(global_start, 0);
        // Make s_r and s_w wait for global_start event
        cudaStreamWaitEvent(s_r, global_start, 0);
        cudaStreamWaitEvent(s_w, global_start, 0);

        r_v8<<<blocks, threads, 0, s_r>>>(d_a, d_out);
        w_v8<<<blocks, threads, 0, s_w>>>(d_b);

        // Make default stream wait for BOTH
        cudaEvent_t r_done, w_done;
        cudaEventCreate(&r_done); cudaEventCreate(&w_done);
        cudaEventRecord(r_done, s_r);
        cudaEventRecord(w_done, s_w);
        cudaStreamWaitEvent(0, r_done, 0);
        cudaStreamWaitEvent(0, w_done, 0);
        cudaEventRecord(global_end, 0);
        cudaEventSynchronize(global_end);

        float ms; cudaEventElapsedTime(&ms, global_start, global_end);
        if (ms < best_total) best_total = ms;
        cudaEventDestroy(r_done); cudaEventDestroy(w_done);
    }

    printf("# True wall-clock concurrent R+W (event on default stream, sync both)\n");
    printf("  Total wall: %.3f ms\n", best_total);
    printf("  Data moved: 4 GB read + 4 GB write = 8 GB total\n");
    printf("  Aggregate BW: %.0f GB/s\n", 2.0*bytes/(best_total/1000)/1e9);
    printf("  Compared to: read-alone 7290, write-alone 7300\n");
    printf("  → Aggregate / max(R,W) alone = %.2fx\n",
           (2.0*bytes/(best_total/1000)/1e9) / 7300);
    return 0;
}
