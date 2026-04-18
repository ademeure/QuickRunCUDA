// Stride sweep: each thread writes ONE int4 at base + tid*S
// Find smallest S that hits all 32 FBPAs evenly
#include <cuda_runtime.h>
#include <cstdio>

template<int STRIDE_BYTES>
__global__ void write_stride(int *data, int n_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    // Write at byte offset = tid * STRIDE_BYTES
    int *p = (int*)((char*)data + (size_t)tid * STRIDE_BYTES);
    int v = 0xab;
    asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
        :: "l"(p), "r"(v) : "memory");
}

int main(int argc, char **argv) {
    int stride_id = (argc > 1) ? atoi(argv[1]) : 0;
    cudaSetDevice(0);
    size_t bytes = 8ull * 1024 * 1024 * 1024;  // 8 GB buffer for stride headroom
    int *d; cudaMalloc(&d, bytes);

    // Total threads = 4 GB / stride (so we write 4 GB total at stride S)
    // Stride values: 32 B, 64, 128, 256, 512, 1024, 2048, 4096, 8192
    int strides[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int n_strides = sizeof(strides)/sizeof(int);
    int s = strides[stride_id];

    long total_bytes_target = 4ll * 1024 * 1024 * 1024;
    int n_threads = total_bytes_target / s;
    int threads = 256;
    int blocks = (n_threads + threads - 1) / threads;

    // Run several times for ncu sampling
    for (int i = 0; i < 5; i++) {
        if (s == 32)    write_stride<32   ><<<blocks, threads>>>(d, n_threads);
        else if (s == 64)   write_stride<64   ><<<blocks, threads>>>(d, n_threads);
        else if (s == 128)  write_stride<128  ><<<blocks, threads>>>(d, n_threads);
        else if (s == 256)  write_stride<256  ><<<blocks, threads>>>(d, n_threads);
        else if (s == 512)  write_stride<512  ><<<blocks, threads>>>(d, n_threads);
        else if (s == 1024) write_stride<1024 ><<<blocks, threads>>>(d, n_threads);
        else if (s == 2048) write_stride<2048 ><<<blocks, threads>>>(d, n_threads);
        else if (s == 4096) write_stride<4096 ><<<blocks, threads>>>(d, n_threads);
        else if (s == 8192) write_stride<8192 ><<<blocks, threads>>>(d, n_threads);
        else if (s == 16384) write_stride<16384><<<blocks, threads>>>(d, n_threads);
    }
    cudaDeviceSynchronize();
    printf("Stride=%d B, n_threads=%d, blocks=%d\n", s, n_threads, blocks);
    return 0;
}
