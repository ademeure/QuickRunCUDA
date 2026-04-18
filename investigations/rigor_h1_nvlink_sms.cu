// H1 RIGOR: NVLink BW vs SM count for R and W separately
//
// THEORETICAL: NVLink-5 (NV18) on B300 = 1500 GB/s bidir, 750 GB/s per dir.
// Hypothesis: read can saturate with few SMs (each SM issues many in-flight
// requests via L2); write limited by SM→L2 BW (each SM has ~46 GB/s L1 BW).

#include <cuda_runtime.h>
#include <cstdio>

extern "C" __launch_bounds__(256, 4) __global__ void nvlink_read(int *src, int *out, int blocks_used) {
    if (blockIdx.x >= blocks_used) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *base = src + warp_id * (32 * 1024 / 4);
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
    else if (tid == 0) out[0] = acc;
}

extern "C" __launch_bounds__(256, 4) __global__ void nvlink_write(int *dst, int blocks_used) {
    if (blockIdx.x >= blocks_used) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *base = dst + warp_id * (32 * 1024 / 4);
    int v = tid;
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        int *p = base + (it * 32 + lane) * 8;
        asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
            :: "l"(p), "r"(v) : "memory");
    }
}

int main() {
    int n; cudaGetDeviceCount(&n);
    if (n < 2) { printf("need 2 GPUs\n"); return 1; }
    cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0);

    size_t bytes = 1024ull * 1024 * 1024;  // 1 GB
    int N = bytes / 4;
    int *d1_data;
    cudaSetDevice(1); cudaMalloc(&d1_data, bytes); cudaMemset(d1_data, 0xab, bytes);
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1 << 24);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int max_blocks = bytes / (256 * 1024);  // = 4096 blocks for full coverage
    printf("# NVLink R/W BW vs SM count (max_blocks=%d for 1 GB)\n", max_blocks);
    printf("# blocks  Read GB/s  Write GB/s  R/W ratio\n");

    int sm_counts[] = {1, 4, 8, 16, 32, 64, 128, 148, 296, 592, 1184, 2368, max_blocks};
    for (int b : sm_counts) {
        if (b > max_blocks) b = max_blocks;
        // Read
        for (int i = 0; i < 3; i++) nvlink_read<<<b, 256>>>(d1_data, d_out, b);
        cudaDeviceSynchronize();
        float t_r = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            nvlink_read<<<b, 256>>>(d1_data, d_out, b);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < t_r) t_r = ms;
        }
        size_t read_bytes = (size_t)b * 256 * 1024;  // each block reads 256 KB
        double r_gbs = read_bytes / (t_r/1000) / 1e9;

        // Write
        for (int i = 0; i < 3; i++) nvlink_write<<<b, 256>>>(d1_data, b);
        cudaDeviceSynchronize();
        float t_w = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            nvlink_write<<<b, 256>>>(d1_data, b);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < t_w) t_w = ms;
        }
        size_t write_bytes = (size_t)b * 256 * 1024;
        double w_gbs = write_bytes / (t_w/1000) / 1e9;

        printf("  %5d   %8.1f   %8.1f   %.2f\n", b, r_gbs, w_gbs, r_gbs/w_gbs);
    }
    return 0;
}
