// DRAM Bandwidth Microbenchmark
// Each thread copies one int4 (16 bytes) from A to C = 32 bytes total (16 read + 16 write)
// Usage: ./QuickRunCUDA tests/bench_dram_bw.cu -t 256 -b <N/4/256> -T 100 -P <bytes/1e9> -U "GB/s"

extern "C" __global__ void kernel(const float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int N, int unused_1, int unused_2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 128-bit vectorized load and store for maximum bandwidth
    ((int4*)C)[idx] = ((const int4*)A)[idx];
}
