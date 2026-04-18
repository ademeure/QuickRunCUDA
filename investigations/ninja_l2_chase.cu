// L2 vs HBM pointer-chase latency at varying WS
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__launch_bounds__(32, 1) __global__ void k_chase(const int *p, int *out, int N_iters) {
    int v = 0;
    long t0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    for (int i = 0; i < N_iters; i++) v = p[v];
    long t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) out[blockIdx.x] = (int)(t1 - t0);
    if (v == 0xDEADBEEF) out[1023] = v;
}

int main() {
    cudaSetDevice(0);
    long sizes_mb[] = {1, 8, 32, 64, 79, 100, 128, 256, 1024};
    int n = sizeof(sizes_mb) / sizeof(sizes_mb[0]);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));

    printf("# L2 vs HBM dependent-chase latency (N_iters = full chain visit)\n");
    printf("# WS    n_lines    N_iters   ns/hop  cy/hop\n");
    for (int s_i = 0; s_i < n; s_i++) {
        long bytes = sizes_mb[s_i] * 1024L * 1024L;
        long N_int = bytes / 4;
        long stride = 32;  // 128B sector stride to hit different cache lines
        long n_lines = bytes / 128;
        // Visit every line at least 4 times to ensure L2 fully populated then evicted if WS > L2
        int N_iters = (int)(n_lines * 4);
        if (N_iters > 4000000) N_iters = 4000000;  // cap at 4M to keep runtime sane
        if (N_iters < 10000) N_iters = 10000;

        // Build random permutation through n_lines distinct cache lines
        long *perm = new long[n_lines];
        for (long i = 0; i < n_lines; i++) perm[i] = i;
        srand(0xCAFE);
        for (long i = n_lines - 1; i > 0; i--) {
            long j = ((unsigned long)rand() << 16 | rand()) % (i + 1);
            long tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
        }
        // Build pointer chain: p[perm[i]*stride] = perm[(i+1)%n]*stride
        int *h_data = new int[N_int];
        for (long i = 0; i < N_int; i++) h_data[i] = 0;
        for (long i = 0; i < n_lines; i++) {
            long src = perm[i] * stride;
            long dst = perm[(i + 1) % n_lines] * stride;
            h_data[src] = (int)dst;
        }

        int *d_data; cudaMalloc(&d_data, bytes);
        cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

        // Warmup
        for (int i = 0; i < 3; i++) k_chase<<<1, 32>>>(d_data, d_out, N_iters);
        cudaDeviceSynchronize();
        // Best of 5
        int best_cy = 0x7fffffff;
        for (int i = 0; i < 5; i++) {
            k_chase<<<1, 32>>>(d_data, d_out, N_iters);
            cudaDeviceSynchronize();
            int h_out;
            cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
            if (h_out > 0 && h_out < best_cy) best_cy = h_out;
        }
        double cy_per_hop = (double)best_cy / N_iters;
        double ns_per_hop = cy_per_hop / 2.032;
        printf("  %4ld MB  lines=%-8ld  N=%-8d  %.1f ns  %.1f cy\n",
               sizes_mb[s_i], n_lines, N_iters, ns_per_hop, cy_per_hop);

        delete[] perm;
        delete[] h_data;
        cudaFree(d_data);
    }
    return 0;
}
