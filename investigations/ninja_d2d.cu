// NINJA D2D copy: try to push beyond cudaMemcpyAsync's 3.3 TB/s
// Theoretical: HBM3E concurrent R+W ceiling = 6.68 TB/s = 3.34 TB/s "per direction"
// cudaMemcpyAsync D2D was 3.28 TB/s — already at SoL?

#include <cuda_runtime.h>
#include <cstdio>

// Ninja copy: each warp reads 1 KB from src, writes 1 KB to dst
__launch_bounds__(256, 4) __global__ void copy_ninja(const int *src, int *dst) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    const int *src_p = src + (warp_id * 32 + lane) * 8;
    int *dst_p = dst + (warp_id * 32 + lane) * 8;
    int r0,r1,r2,r3,r4,r5,r6,r7;
    asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
        : "l"(src_p));
    asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "l"(dst_p), "r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(r4),"r"(r5),"r"(r6),"r"(r7) : "memory");
}

// 2 KB per warp variant
__launch_bounds__(256, 4) __global__ void copy_2kb(const int *src, int *dst) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    #pragma unroll
    for (int it = 0; it < 2; it++) {
        const int *src_p = src + ((warp_id * 2 + it) * 32 + lane) * 8;
        int *dst_p = dst + ((warp_id * 2 + it) * 32 + lane) * 8;
        int r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "l"(src_p));
        asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
            :: "l"(dst_p), "r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(r4),"r"(r5),"r"(r6),"r"(r7) : "memory");
    }
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 4ull * 1024 * 1024 * 1024;  // 4 GB src + 4 GB dst = 8 GB total HBM traffic
    int *d_src, *d_dst;
    cudaMalloc(&d_src, bytes); cudaMalloc(&d_dst, bytes);
    cudaMemset(d_src, 0xab, bytes);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](auto launch, const char* label) {
        for (int i = 0; i < 5; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 30; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        // bytes_per_dir = bytes (just one read, one write = same byte count)
        // But total HBM traffic = 2 * bytes (R + W)
        size_t total_bytes = bytes * 2;
        double agg = total_bytes / (best/1000) / 1e9;
        double per_dir = bytes / (best/1000) / 1e9;
        printf("  %s: %.4f ms = R+W aggregate %.0f GB/s = per-dir %.0f GB/s = %.2f%% of HBM peak\n",
               label, best, agg, per_dir, agg/7672*100);
    };

    int blocks_1kb = bytes / (256 * 32);
    int blocks_2kb = bytes / (256 * 64);

    bench([&]{ cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice); }, "cudaMemcpyAsync D2D");
    bench([&]{ copy_ninja<<<blocks_1kb, 256>>>(d_src, d_dst); }, "copy_ninja 1KB/warp");
    bench([&]{ copy_2kb<<<blocks_2kb, 256>>>(d_src, d_dst); }, "copy_2kb 2KB/warp  ");

    return 0;
}
