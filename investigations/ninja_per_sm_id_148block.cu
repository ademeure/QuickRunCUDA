// Use exactly 148 blocks - one per SM, no atomicCAS bias
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
__global__ void per_sm_lat(uint32_t *p, uint64_t *out) {
    if (threadIdx.x != 0) return;
    int sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    uint32_t cur = 0;
    for (int i = 0; i < 100; i++) {
        uint32_t v;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p + cur));
        cur = v;
    }
    long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < 200; i++) {
        uint32_t v;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p + cur));
        cur = v;
    }
    long long t1 = clock64();
    out[sm_id*2] = (uint64_t)(t1 - t0);
    out[sm_id*2 + 1] = cur;
}
int main(int argc, char**argv) {
    long offset = (argc > 1) ? atol(argv[1]) : 0;
    cudaSetDevice(0);
    size_t bytes = 2ULL * 1024 * 1024 * 1024;
    uint32_t *d_p; cudaMalloc(&d_p, bytes);
    cudaMemset(d_p, 0, bytes);
    uint64_t *d_out; cudaMalloc(&d_out, 200 * 16);
    cudaMemset(d_out, 0, 200 * 16);
    per_sm_lat<<<148, 32>>>((uint32_t*)((char*)d_p + offset), d_out);
    cudaDeviceSynchronize();
    uint64_t h[200 * 2];
    cudaMemcpy(h, d_out, 200 * 16, cudaMemcpyDeviceToHost);
    int n = 0; long long mn = 1LL<<60, mx = 0;
    long lat[200]; int sm_seen[200];
    for (int i = 0; i < 200; i++) {
        if (h[i*2] > 0) { sm_seen[n] = i; lat[n] = h[i*2]; n++; if ((long long)h[i*2]<mn) mn=h[i*2]; if ((long long)h[i*2]>mx) mx=h[i*2]; }
    }
    long mid = (mn + mx) / 2;
    printf("offset=%-12ld n=%d Min=%.1f Mean=%.1f Max=%.1f cy/load near SMs:", offset, n, (double)mn/200, (double)((mn+mx)/2)/200, (double)mx/200);
    for (int i = 0; i < n; i++) {
        if (lat[i] < mid) printf(" %d", sm_seen[i]);
    }
    printf("\n");
    return 0;
}
