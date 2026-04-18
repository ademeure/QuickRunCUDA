// Write a large amount of data to addresses that, by hypothesis,
// all map to ONE FBPA. Verify via ncu min/max.
#include <cuda_runtime.h>
#include <cstdio>

// Thread tid writes to base + interleave_block * granularity
// where (interleave_block & 31) == target_fbpa_id (assuming 32-FBPA mod interleave)
template<int GRAN_BITS>
__global__ void write_target_fbpa(int *base, int n_threads, int target_id, int n_fbpas) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    long block_id = (long)tid * n_fbpas + target_id;  // ensures (block_id & (n_fbpas-1)) == target_id
    long offset = block_id << GRAN_BITS;  // multiply by granularity (in bytes)
    int *p = (int*)((char*)base + offset);
    int v = 0xab;
    asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
        :: "l"(p), "r"(v) : "memory");
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    int gran_bits = (argc > 1) ? atoi(argv[1]) : 8;  // 8 = 256-byte interleave hypothesis
    int target_id = (argc > 2) ? atoi(argv[2]) : 0;
    int n_fbpas = 32;

    long target_bytes = 256ll * 1024 * 1024;  // 256 MB to touch
    long granularity = 1LL << gran_bits;
    long buf_bytes = target_bytes * n_fbpas;  // need granularity * n_threads * n_fbpas
    if (buf_bytes > 16ll * 1024 * 1024 * 1024) buf_bytes = 16ll * 1024 * 1024 * 1024;
    int *d; cudaMalloc(&d, buf_bytes);

    int n_threads = target_bytes / granularity;  // hits target_bytes worth of ONE FBPA (if hypothesis correct)
    int threads = 256;
    int blocks = (n_threads + threads - 1) / threads;
    if (n_threads <= 0 || blocks <= 0) { printf("Bad config\n"); return 1; }

    for (int i = 0; i < 3; i++) {
        if (gran_bits == 6) write_target_fbpa<6><<<blocks, threads>>>(d, n_threads, target_id, n_fbpas);
        else if (gran_bits == 7) write_target_fbpa<7><<<blocks, threads>>>(d, n_threads, target_id, n_fbpas);
        else if (gran_bits == 8) write_target_fbpa<8><<<blocks, threads>>>(d, n_threads, target_id, n_fbpas);
        else if (gran_bits == 9) write_target_fbpa<9><<<blocks, threads>>>(d, n_threads, target_id, n_fbpas);
        else if (gran_bits == 10) write_target_fbpa<10><<<blocks, threads>>>(d, n_threads, target_id, n_fbpas);
        else if (gran_bits == 11) write_target_fbpa<11><<<blocks, threads>>>(d, n_threads, target_id, n_fbpas);
        else if (gran_bits == 12) write_target_fbpa<12><<<blocks, threads>>>(d, n_threads, target_id, n_fbpas);
    }
    cudaDeviceSynchronize();
    printf("granularity_bits=%d target=%d n_threads=%d\n", gran_bits, target_id, n_threads);
    return 0;
}
