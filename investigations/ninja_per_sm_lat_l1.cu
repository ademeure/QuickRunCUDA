// Per-SM single-line latency: SELF-CHAIN where p[0] = 0, so cur = p[cur] = 0 forever
// Each load is forced sequential dep
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void per_sm_chain(uint32_t *p, uint64_t *out, int max_sms) {
    if (threadIdx.x != 0) return;
    int sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    uint32_t cur = 0;
    // warmup
    for (int i = 0; i < 100; i++) {
        uint32_t v;
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p + cur));
        cur = v;
    }
    long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < 200; i++) {
        uint32_t v;
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p + cur));
        cur = v;
    }
    long long t1 = clock64();
    if (sm_id < max_sms) {
        atomicCAS((unsigned long long*)&out[sm_id*2], 0ULL, (unsigned long long)(t1 - t0));
        out[sm_id*2 + 1] = cur;
    }
}

int main(int argc, char**argv) {
    long offset = (argc > 1) ? atol(argv[1]) : 0;
    cudaSetDevice(0);
    size_t bytes = 2ULL * 1024 * 1024 * 1024;
    uint32_t *d_p; cudaMalloc(&d_p, bytes);
    cudaMemset(d_p, 0, bytes);  // all zeros, so p[0] = 0 -> cur stays 0
    int max_sms = 200;
    uint64_t *d_out; cudaMalloc(&d_out, max_sms * 16);
    cudaMemset(d_out, 0, max_sms * 16);
    per_sm_chain<<<2000, 32>>>((uint32_t*)((char*)d_p + offset), d_out, max_sms);
    cudaDeviceSynchronize();
    uint64_t h[200 * 2];
    cudaMemcpy(h, d_out, 200 * 16, cudaMemcpyDeviceToHost);
    int n = 0; long long mn = 1LL<<60, mx = 0, total = 0;
    long lat[200];
    for (int i = 0; i < max_sms; i++) {
        if (h[i*2] > 0) { lat[n++] = h[i*2]; if ((long long)h[i*2]<mn) mn=h[i*2]; if ((long long)h[i*2]>mx) mx=h[i*2]; total += h[i*2]; }
    }
    if (!n) { printf("No SMs\n"); return 1; }
    long mid = (mn + mx) / 2;
    int near = 0, far = 0; long sn = 0, sf = 0;
    for (int i = 0; i < n; i++) { if (lat[i] < mid) { near++; sn += lat[i]; } else { far++; sf += lat[i]; } }
    printf("offset=%-12ld  Min=%.1f Mean=%.1f Max=%.1f cy/load  Range=%.2fx  Near=%d (%.1f cy=%.1f ns)  Far=%d (%.1f cy=%.1f ns)\n",
        offset,
        (double)mn/200, (double)total/n/200, (double)mx/200, (double)mx/mn,
        near, near?(double)sn/near/200:0, near?(double)sn/near/200/2.032:0,
        far,  far?(double)sf/far/200:0,   far?(double)sf/far/200/2.032:0);
    return 0;
}
